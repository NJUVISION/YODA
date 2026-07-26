from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch
import torch.nn as nn
# === LoRA 模块定义 ===
class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=4):
        super().__init__()
        self.conv = conv
        in_channels, out_channels = conv.in_channels, conv.out_channels
        self.lora_A = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_B(self.lora_A(x))

# === LoRA 注入器 ===
def inject_lora(module, rank=4):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank))
        else:
            inject_lora(child, rank)

def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    if "model" in ckpt:
        ckpt = ckpt["model"]

    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def get_alpha_bar_schedule2(config):
    num_steps = config["model"]["params"]["timesteps"]
    beta_start = config["model"]["params"]["linear_start"]
    beta_end = config["model"]["params"]["linear_end"]

    # 构造 beta 调度（线性）
    betas = torch.linspace(beta_start, beta_end, num_steps)

    # 计算 alpha
    alphas = 1.0 - betas

    # 累积乘积得到 alpha_bar
    alpha_bars = torch.cumprod(alphas, dim=0)

    return alpha_bars, betas, alphas