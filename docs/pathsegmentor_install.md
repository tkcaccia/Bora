# PathSegmentor installation for Bora

Bora uses the official PathSegmentor v1.0.0 code and checkpoint. The verified
host configuration is Ubuntu, Python 3.10, an RTX 5060 Ti, NVIDIA driver
595.84, CUDA toolkit 13.0, and PyTorch 2.14.0+cu130. Keep this environment
separate from MedSAM because PathSegmentor pins older Python packages and
compiles two CUDA extensions.

## 1. Clone the published source

```bash
git clone https://github.com/zhi-xuan-chen/PathSegmentor.git
git -C PathSegmentor checkout e67763ab835b3ef184cb6791eb2cd059ec8b9f9a
git -C PathSegmentor apply --unidiff-zero \
  /path/to/Bora/compat/pathsegmentor-modern-cuda.patch
```

The Bora patch updates two deprecated PyTorch C++ dispatch calls and makes MPI
optional for single-process inference. It does not change model weights or
inference mathematics.

## 2. Create an isolated Python environment

Create a Python 3.10 virtual environment, activate it, and install a PyTorch
wheel matching the locally installed CUDA toolkit. For CUDA 13.0:

```bash
python3.10 -m venv .venv-pathsegmentor
source .venv-pathsegmentor/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Install the packages in the official `requirements.txt`. Its first entry is a
custom Detectron2 Git dependency; build it with isolation disabled and with
`CUDA_HOME` set to the toolkit matching PyTorch. Install every other official
requirement from a filtered temporary file.

```bash
CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=12.0 \
  python -m pip install --no-build-isolation \
  'detectron2 @ git+https://github.com/MaureenZOU/detectron2-xyz.git@42121d75e10d9f858f3a91b6a39f5722c02868f0'
grep -v '^detectron2' PathSegmentor/requirements.txt \
  > /tmp/pathsegmentor-requirements.txt
python -m pip install -r /tmp/pathsegmentor-requirements.txt
```

Compile PathSegmentor's deformable-attention extension:

```bash
cd PathSegmentor/modeling/vision/encoder/ops
CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=12.0 FORCE_CUDA=1 \
  python -m pip install . --no-build-isolation
```

Change `TORCH_CUDA_ARCH_LIST` for another GPU. The RTX 5060 Ti uses `12.0`.

## 3. Checkpoint and smoke test

Download `model_state_dict.pt` from the link in the official PathSegmentor
README. The checkpoint used for Bora validation had SHA-256:

```text
008c99b65f1b2a4f130300b298ef5e1425fecdfb6224a43b89ab729eb46763ba
```

PathSegmentor downloads its language encoder on first use, so the first model
load requires network access. Run Bora with a JSON prompt for every non-zero
label:

```bash
bora refine image.ome.tif mask.tif \
  --output refined.ome.tif --geojson refined.geojson \
  --backend pathsegmentor --repo-dir /path/to/PathSegmentor \
  --checkpoint /path/to/model_state_dict.pt --label-map labels.json \
  --device cuda
```

Numeric clusters without semantic names should use Bora's watershed backend.
Do not give every cluster the same PathSegmentor prompt: identical predictions
would compete and could corrupt cluster identity.
