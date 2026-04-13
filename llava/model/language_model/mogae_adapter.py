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
    ):
        super().__init__()
        if router_input_dim is None:
            raise ValueError("router_input_dim must be provided for MoGAE")

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
        self.router_mlp = nn.Sequential(
            nn.Linear(router_input_dim, router_input_dim),
            nn.GELU(),
            nn.Linear(router_input_dim, num_adapters),
        )
        self.num_adapters = num_adapters
        self.router_embedding = None
        self.sparse = sparse
        self.routing_outcome = None
        self.lb_loss = None

    def forward(self, x, add_residual=True, residual=None):
        if self.router_embedding is None:
            raise RuntimeError("MoGAE router embedding is not set")

        router_embedding = self.router_embedding.to(device=x.device, dtype=x.dtype)
        logits = self.router_mlp(router_embedding)
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
            output = output + adapter(x, add_residual, residual) * ratios[:, idx].view(-1, 1, 1)
        return output


def set_router_embedding_mogae(model: nn.Module, router_embedding: torch.Tensor):
    for child in model.children():
        if isinstance(child, MoGAEAdapterRouter):
            child.router_embedding = router_embedding
        elif len(list(child.children())) != 0:
            set_router_embedding_mogae(child, router_embedding)
