# Setup Guide (new machine checklist)

This repo has two mostly-independent pieces:

* **Lane Digitize Tool** — manual LAS upload + lane digitizing (Electron app / webapp)
* **Lane3D Training** — Cylinder3D model training pipeline (see [`training/README.md`](training/README.md))

Both need Node.js (to build the Potree viewer) and PDAL/GDAL; the training
pipeline additionally needs a conda env and the mmdet3d/torch stack.

## 1. System packages

```bash
sudo apt install nodejs npm gdal-bin
```

PDAL is deliberately **not** installed via apt/pip here — see step 3.

## 2. Node / npm

Builds the Potree viewer (`postinstall` runs the build automatically):

```bash
npm install
```

This also downloads a prebuilt Electron binary (~100MB) as part of installing the
`electron` dependency - needs network access, and can take a few minutes.

## 3. Python + PDAL

`pip install pdal` is broken for every version from 2.3.x through the current
3.5.x: the sdist's `setup.py` only looks for `CMakeLists.txt` at the sdist root,
but it actually ships under `pdal/CMakeLists.txt`, so the build "succeeds" but
silently skips compiling the `libpdalpython` C++ extension
(`ImportError: cannot import name 'libpdalpython'` the moment you `import pdal`).
Found and confirmed on 2026-07-14 — install PDAL from conda-forge instead, which
ships a real prebuilt binary:

```bash
conda create -n lane3d --override-channels -c conda-forge python=3.10 pdal python-pdal gdal
conda activate lane3d
```

(No conda/miniconda installed yet? <https://docs.conda.io/projects/miniconda/en/latest/>,
no sudo needed. After installing, run `conda init bash` — or your shell — once so
`conda activate` works in new terminals.)

If this machine already has other `pip install --user ...` packages, they can leak
into the conda env ahead of its own packages (caused a real numpy ABI crash during
this setup) — see the isolation fix in
[`training/README.md`](training/README.md#1-conda-environment).

**3a. Lane Digitize Tool webapp only** (run with `lane3d` active, same as step 3):

```bash
pip install fastapi uvicorn python-multipart
```

**3b. Lane3D training pipeline** (additional, on top of 3a):

```bash
pip install -r training/requirements.txt
```

The mmdet3d/mmcv/PyTorch stack (only needed for actual training/inference, not
data prep) is covered separately in
[`training/README.md`](training/README.md#2-python-dependencies).

**Important — this env must be active in the terminal you launch the app from.**
`electron/main.js` starts the backend with a plain `spawn('python3', ...)`, which
resolves `python3` from whatever's first on `PATH` at that moment - not a fixed
path. If you run `npm run desktop` from a terminal where `lane3d` isn't activated,
Electron opens fine but the backend crashes immediately
(`ModuleNotFoundError: No module named 'fastapi'`) and the window shows
"Server did not become ready" / `ERR_CONNECTION_REFUSED`. Always
`conda activate lane3d` in the same terminal first.

## 4. Data NOT tracked in git — copy manually to a new machine

| What | Used by | Notes |
|---|---|---|
| Raw LAS corridor files | `training/prepare_dataset.py`, `convert_all.sh` | path passed as a CLI arg, any location |
| HD map SHP export (`B2_SURFACELINEMARK.shp`, `B3_SURFACEMARK.shp`, ...) | `training/prepare_dataset.py` | path passed as a CLI arg, any location |
| `training/data/`, `training/work_dirs/`, `training/logs/` | training pipeline | gitignored (large, regenerable) — rsync/scp from the old machine, or regenerate with `convert_all.sh` |
| `webapp/hdmap/` | local SHP-overlay testing in the webapp | gitignored, optional |

## 5. Verify

```bash
conda activate lane3d      # must be active in THIS terminal before either check below
npm run desktop                                    # Lane Digitize Tool - opens a window,
                                                    # no "Server did not become ready" error
python3 training/prepare_dataset.py --help         # training - prints usage, no ImportError
```
