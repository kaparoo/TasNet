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

    def save(self, path: str):
        with open(path, "w", encoding="utf8") as f:
            f.write('\n'.join(f"{k}={v}" for k, v
                    in self.get_config().items()))

    @staticmethod
    def load(path: str):
        def convert_int(value):
            try:
                return int(value)
            except:
                pass
            return value

        def convert_bool(value):
            if value == 'True':
                return True
            elif value == 'False':
                return False
            else:
                return value

        def convert_tup(tup):
            if tup[0] == 'causal':
                return (tup[0], convert_bool(tup[1]))
            else:
                return (tup[0], convert_int(tup[1]))

        with open(path, "r", encoding="utf8") as f:
            d = dict(convert_tup(line.strip().split('='))
                     for line in f.readlines())
            return TasNetParam(**d)
