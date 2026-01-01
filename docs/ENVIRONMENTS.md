# Environments

TreeBench has one current environment for dataset construction, validation,
and the shared 32-metric evaluator, plus three legacy compatibility profiles
for the reported baselines.

Create the core environment with:

```bash
conda env create -f environments/core.yml
conda activate treebench-core
python -m pip install -e .
pytest
```

The baseline registry assigns every model to one of
`legacy_repkpu`, `legacy_pointr`, or `legacy_pudm`. These files record the
Python/PyTorch/CUDA-era combinations used for the paper runs; they intentionally
exclude locally built CUDA extensions. After materializing a baseline, build
its Chamfer, PointNet++, KNN, EMD, or pointops extension from that pinned
upstream checkout as described by its retained upstream README and license.
Never copy a prebuilt `.so` from another machine.

PyTorch wheels with CUDA build suffixes may require the matching official
PyTorch wheel index. PyTorch3D 0.6.1 is additionally required by the PUDM
profile and must match PyTorch 1.9.1/CUDA 11.1. Modern drivers can run these
older CUDA runtimes, but compilation still requires a compatible compiler and
CUDA toolkit. Record compiler, driver, GPU, and generated-extension hashes in
the run manifest.

These environment profiles document the executed experiments; they are not a
claim that the 16 baselines share one compute budget or one modern software
stack.

