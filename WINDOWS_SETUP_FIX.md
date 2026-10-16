# Windows setup fix for Python 3.14 / NumPy install failure

The original dependency file used open-ended minimum versions (`>=`). In the supplied install log, pip backtracked to NumPy 1.26.4 and tried to compile it from source. NumPy 1.26.4 supports Python 3.9–3.12, not Python 3.14, so use Python 3.12 for this reproducible setup.

## 1. Install Python 3.12

Install the latest Python 3.12.x 64-bit Windows release from https://www.python.org/downloads/windows/. Keep Python 3.14 installed if you need it for other projects.

Open a new PowerShell terminal and check:

```powershell
py -0p
py -3.12 --version
```

## 2. Re-create the virtual environment

Run these commands from the project root. They delete only the project's existing `.venv` folder, not your source code.

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If `deactivate` says the command is not recognized, continue with the next command.

## 3. Install CPU PyTorch and the pinned dependencies

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The first command installs CPU-only PyTorch and its matching torchvision version; the second installs the remaining dependencies and verifies the same pinned torch versions.

## 4. Verify, train, evaluate, and launch

```powershell
python -c "import sys, torch, torchvision, numpy; print(sys.version); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m pytest -q
python -m src.train --epochs 5 --batch-size 32 --num-workers 0
python -m src.evaluate --checkpoint artifacts/best_model.pt --ood-datasets svhn cifar100
streamlit run app.py
```

If PowerShell refuses to activate the venv, use this for the current terminal only and then activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Training and evaluation download datasets and pretrained weights, so keep an internet connection available.
