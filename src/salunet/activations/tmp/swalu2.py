class SWALU(nn.Module):
    r"""SWALU(x) = 0.5*x*(1 + SALU(x))."""

    def __init__(
        self,
        a_init: float = 1.0,
        b_init: float = 1.0,
        name: str = "",
    ) -> None:
        super().__init__()

        self.salu = SALU(
            a_init=a_init,
            b_init=b_init,
            name=f"{name}.salu" if name else "swalu.salu",
        )
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + self.salu(x))

    @torch.no_grad()
    def get_geometry(self) -> Dict[str, torch.Tensor]:
        return self.salu.get_geometry()

    @torch.no_grad()
    def get_params_for_logging(
        self,
        prefix: str = "",
    ) -> Dict[str, float]:
        base = prefix or self.name or "swalu"
        return self.salu.get_params_for_logging(base)