# Installation

The original experiments use Python 3.8.20, PyTorch 1.9.1+cu111,
torchvision 0.10.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0,
mmsegmentation 0.14.1, mmdet3d 0.17.1, spconv-cu111 2.1.25,
and torch-scatter 2.0.8. The source package was checked in that existing
environment. A fresh installation and full dataset inference were not rerun
as part of packaging.

Use an NVIDIA GPU and a compatible driver. Compiling extensions may require
the CUDA 11.1 toolkit and a compatible C++ compiler. This is the legacy
OpenMMLab stack; current mmdet3d/MMEngine versions use different APIs.

```bash
conda create -n umpocc python=3.8 -y
conda activate umpocc
python -m pip install 'numpy==1.23.5'
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install mmdet==2.14.0 mmsegmentation==0.14.1
python -m pip install spconv-cu111==2.1.25
python -m pip install torch-scatter==2.0.8 \
  -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

Install mmdet3d in a separate directory outside this repository:

```bash
git clone --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git
python -m pip install -v -e ./mmdetection3d
```

Then return to the UMPOcc root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python tools/train.py --help
python tools/test_ood_fast.py --help
```

The shell entry points set `PYTHONPATH` automatically. They use `python` from
the active environment; set `PYTHON_BIN` to select a different interpreter.
The training entry point retains the existing YAPF/TensorBoard compatibility
handling from the research code.

SPIdepth preprocessing additionally needs a separate SPIdepth checkout, its
own dependencies, and its SQL weights. The path is passed with
`--spidepth-root`; it is not an implicit dependency of model evaluation.
