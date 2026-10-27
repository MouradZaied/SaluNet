import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

'''
class Dynamic_salu(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=1.0, shift_init_value=0.00025, eps = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.eps = eps
        # Use log parameterization for positive constraints
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init_value, dtype=torch.float32)))
        self.log_shift = nn.Parameter(torch.log(torch.tensor(shift_init_value, dtype=torch.float32)))
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        alpha = torch.exp(self.log_alpha) + self.eps
        shift = torch.exp(self.log_shift) + self.eps
        x = x.float()

        # erf-like activation: (alpha * x) / sqrt(1 + alpha*shift*x^2)
        x_squared = x * x
        denom = torch.sqrt(1 + alpha * shift * x_squared + self.eps)
        x = (alpha * x) / denom
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x
            

    def extra_repr(self):
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha).item()
            shift = torch.exp(self.log_shift).item()
        return (f"normalized_shape={self.normalized_shape}, "
                f"alpha={alpha:.4f}, shift={shift:.4f}, "
                f"channels_last={self.channels_last}")

 


def convert_ln_to_salu(module, alpha_init_value=0.5, shift_init_value=0.0):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = Dynamic_salu(module.normalized_shape, not isinstance(module, LayerNorm2d), alpha_init_value, shift_init_value)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_salu(child, alpha_init_value, shift_init_value))
    del module
    return module_output 
'''


class SALU(nn.Module):
    def __init__(self, a_init=1.0, b_init=0.1, eps=1e-6):
        super().__init__()
        # Log-param trick = stabilité + positivité de a et b
        self.loga = nn.Parameter(torch.log(torch.tensor(a_init, dtype=torch.float32)))
        self.logb = nn.Parameter(torch.log(torch.tensor(b_init, dtype=torch.float32)))
        self.eps = eps

    def forward(self, x):
        # Applique la fonction SALU sans paramètres appris (purement déterministe)
        x = x.float()
        a = torch.exp(self.loga) + self.eps
        b = torch.exp(self.logb) + self.eps
        denom = torch.sqrt(1 + a * b * x**2 + self.eps)
        return (a * x) / denom


class SWALU(nn.Module):
    """
    SWALU : Smooth version of ReLU using SALU
    SWALU(x) = 0.5 * x * (1 + SALU(x))
    Utilisée comme fonction d'activation (remplace GELU).
    """
    def __init__(self, a=0.5, b=0.5, eps=1e-6):
        super().__init__()
        # a et b sont des paramètres appris, initialisés à 1
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        # Garantir a et b positifs
        a = torch.abs(self.a) + self.eps
        b = torch.abs(self.b) + self.eps
        x = x.float()

        denom = torch.sqrt(1 + a * b * x**2 + self.eps)
        salu_x = (a * x) / denom
        return 0.5 * x * (1 + salu_x) 



# ========== Remplacement layernorm function -> SALU ==========
def convert_ln_to_salu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            # Créer une instance SALU avec les mêmes paramètres si nécessaire
            # (ignorer affine, car SALU n'a pas de poids)
            new_salu = SALU(a_init=1.0, b_init=0.1, eps=1e-6)
            setattr(module, name, new_salu)
        else:
            convert_ln_to_salu(child)
    return module

    
# ========== Remplacement GELU function -> SWALU ==========
def convert_gelu_to_swalu(module, a=1.0, b=1.0, clamp_value=10.0, eps=1e-6):
    """
    Parcourt récursivement le modèle et remplace toutes les instances de GELU
    par SWALU.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, SWALU(a=a, b=b,  eps=eps))
        else:
            convert_gelu_to_swalu(child, a=a, b=b, eps=eps)
    return module
            