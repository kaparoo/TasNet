class TasNetParam:

    __slots__ = 'K', 'C', 'L', 'N', 'H', 'causal'

    def __init__(self,
                 K: int = 20,
                 C: int = 4,
                 L: int = 40,
                 N: int = 500,
                 H: int = 1000,
                 causal: bool = False):
        self.K = K
        self.C = C
        self.L = L
        self.N = N
        self.H = H
        self.causal = causal

    def get_config(self) -> dict:
        return {'K': self.K,
                'C': self.C,
                'L': self.L,
                'N': self.N,
                'H': self.H,
                'causal': self.causal}
