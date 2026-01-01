from .CD import (cd, fscore)

# EMD is not used by FSC/TreeBench and its eager import triggers a local CUDA
# compilation on every process. Keep it optional so the official FSC model and
# Chamfer metric can run without this unrelated extension.
try:
    from .EMD import emd
except Exception:
    emd = None

__all__ = [
    'cd', 'fscore', 'emd',
]
