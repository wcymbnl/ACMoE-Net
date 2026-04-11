import math

import torch
import torch.nn.functional as F
from torch import nn


def top1_gate(logits):
    y_soft = logits.softmax(dim=-1)
    gate, index = y_soft.max(dim=-1, keepdim=True)
    lb_loss = gate.sum(dim=-1).mean()
    y_hard = torch.zeros_like(logits, device=gate.device).scatter_(-1, index, 1.0)
    return y_hard - y_soft.detach() + y_soft, lb_loss


class Adapter(nn.Module):
    def __init__(
        self,
        d_model,
        bottleneck=64,
        dropout=0.0,
        init_option="lora",
        adapter_scalar="learnable_scalar",
        adapter_layernorm_option="none",
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in {"in", "out"}:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        if init_option != "lora":
            raise NotImplementedError(f"Unsupported init option: {init_option}")

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        return up + residual if add_residual else up


class MoGAEAdapterRouter(nn.Module):
    def __init__(
        self,
        d_model,
        bottleneck=64,
        dropout=0.0,
        init_option="lora",
        adapter_scalar="learnable_scalar",
        adapter_layernorm_option="none",
        num_adapters=4,
        router_input_dim=None,
        sparse=True,
        router_type="sentence",
    ):
        super().__init__()
        if router_type not in {"token", "sentence", "token_sentence"}:
            raise ValueError(f"Unsupported router_type: {router_type}")

        self.adapters = nn.ModuleList(
            [
                Adapter(
                    d_model=d_model,
                    bottleneck=bottleneck,
                    dropout=dropout,
                    init_option=init_option,
                    adapter_scalar=adapter_scalar,
                    adapter_layernorm_option=adapter_layernorm_option,
                )
                for _ in range(num_adapters)
            ]
        )
        default_router_dim = {
            "token": d_model,
            "sentence": d_model,
            "token_sentence": d_model * 2,
        }[router_type]
        self.router_input_dim = router_input_dim or default_router_dim
        if self.router_input_dim != default_router_dim:
            self.router_proj = nn.Linear(default_router_dim, self.router_input_dim)
        else:
            self.router_proj = None
        self.router_mlp = nn.Sequential(
            nn.Linear(self.router_input_dim, self.router_input_dim),
            nn.GELU(),
            nn.Linear(self.router_input_dim, num_adapters),
        )
        self.num_adapters = num_adapters
        self.router_embedding = None
        self.sparse = sparse
        self.router_type = router_type
        self.routing_outcome = None
        self.lb_loss = None

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _build_router_input(self, x: torch.Tensor,
                            router_text_mask: torch.Tensor | None) -> torch.Tensor:
        if self.router_type == "token":
            router_input = x
        elif self.router_type == "sentence":
            if router_text_mask is None:
                if self.router_embedding is None:
                    raise RuntimeError("router_text_mask is required for sentence routing")
                return self.router_embedding.to(device=x.device, dtype=x.dtype)
            router_input = self._masked_mean_pool(x, router_text_mask)
        else:
            if router_text_mask is None:
                raise RuntimeError("router_text_mask is required for token_sentence routing")
            sentence_embedding = self._masked_mean_pool(x, router_text_mask)
            sentence_embedding = sentence_embedding.unsqueeze(1).expand(-1, x.shape[1], -1)
            router_input = torch.cat([x, sentence_embedding], dim=-1)

        if self.router_proj is not None:
            router_input = self.router_proj(router_input)
        return router_input

    def forward(self, x, router_text_mask=None, add_residual=True, residual=None):
        router_input = self._build_router_input(x, router_text_mask)
        logits = self.router_mlp(router_input)
        if self.sparse:
            ratios, self.lb_loss = top1_gate(logits)
        else:
            ratios = torch.softmax(logits, dim=-1)
            self.lb_loss = None

        self.routing_outcome = ratios

        if x.shape[0] % ratios.shape[0] != 0:
            raise ValueError(
                f"x batch size {x.shape[0]} must be divisible by router batch size {ratios.shape[0]}"
            )
        num_beams = x.shape[0] // ratios.shape[0]
        ratios = ratios.repeat_interleave(num_beams, dim=0)

        output = 0
        for idx, adapter in enumerate(self.adapters):
            if ratios.dim() == 2:
                adapter_ratio = ratios[:, idx].view(-1, 1, 1)
            else:
                adapter_ratio = ratios[:, :, idx].unsqueeze(-1)
            output = output + adapter(x, add_residual, residual) * adapter_ratio
        return output


def set_router_embedding_mogae(model: nn.Module, router_embedding: torch.Tensor):
    for child in model.children():
        if isinstance(child, MoGAEAdapterRouter):
            child.router_embedding = router_embedding
        elif len(list(child.children())) != 0:
            set_router_embedding_mogae(child, router_embedding)
