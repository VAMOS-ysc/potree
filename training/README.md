# Lane3D Training Pipeline

Trains a [Cylinder3D](https://github.com/xinge008/Cylinder3D) point-cloud segmentation
model (background / lane / crosswalk) on ground-classified LiDAR corridor scans, using
HD map SHP layers as ground truth. Predictions can be turned back into vector
Shapefiles and loaded into the Potree viewer the same way the HD map overlays are.

Pipeline: `prepare_dataset.py` → `make_info_pkl.py` → train (mmdet3d) →
`run_inference.py` → `vectorize_predictions.py`.

For the full new-machine checklist (system packages, npm, and the raw LAS/HD map
data that isn't tracked in git), see [`../SETUP.md`](../SETUP.md). This doc covers
just the Python/conda side in detail.

## 1. Conda environment

A conda env is not optional here (unlike most Python projects) - `pip install pdal`
is broken (see below), and mmcv/mmdet3d's version matching is much easier to get
right through conda-forge/`mim` than by hand. Miniconda install (no sudo needed):
<https://docs.conda.io/projects/miniconda/en/latest/>.

```bash
conda create -n lane3d --override-channels -c conda-forge python=3.10
conda activate lane3d
```

If this machine already has any `pip install --user ...` packages (check with
`pip list --user`), they leak into every conda env's `import` resolution ahead of
the env's own packages, which caused a real `numpy.dtype size changed, may
indicate binary incompatibility` crash in scikit-image during setup. Fix once per
env by making activation ignore user-site packages:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
echo 'export PYTHONNOUSERSITE=1' > "$CONDA_PREFIX/etc/conda/activate.d/pythonnousersite.sh"
echo 'unset PYTHONNOUSERSITE' > "$CONDA_PREFIX/etc/conda/deactivate.d/pythonnousersite.sh"
conda deactivate && conda activate lane3d   # re-activate to pick it up
```

## 2. Python dependencies

**pdal is a special case**: `pip install pdal` (any version from 2.3.x through the
latest 3.5.x) builds without error but silently produces a broken package - its
`setup.py` only looks for `CMakeLists.txt` at the sdist root, but it actually ships
nested under `pdal/CMakeLists.txt`, so the build quietly skips compiling the
`libpdalpython` C++ extension and installs pure-Python stub files instead
(`ImportError: cannot import name 'libpdalpython'` the moment you `import pdal`).
Install it from conda-forge instead, which ships a real prebuilt binary:

```bash
conda install -y -c conda-forge pdal python-pdal gdal
```

Then the rest, straightforward pip installs, via [`requirements.txt`](requirements.txt):

```bash
pip install -r training/requirements.txt
```

**Training / inference** (needs a CUDA GPU): install PyTorch matching your CUDA
version, then the OpenMMLab stack via `mim`:

```bash
pip install openmim
mim install mmengine mmcv mmdet mmdet3d
pip install spconv-cuXXX   # match your CUDA version, e.g. spconv-cu120
```

## 3. Directory layout

```
training/
  configs/cylinder3d_lane3d.py   # model + dataset config (paths resolve relative
                                  # to this repo automatically, no editing needed)
  prepare_dataset.py             # raw LAS + HD map SHP -> SemanticKITTI-format tiles
  make_info_pkl.py                # tiles -> train/val .pkl info files for mmdet3d
  run_inference.py                # trained checkpoint -> predicted GeoJSON points
  vectorize_predictions.py        # predicted points -> lane/crosswalk Shapefile
  convert_all.sh                  # runs prepare_dataset.py over every *.las in a dir
  data/        (gitignored)       # generated tiles - regenerate with convert_all.sh
  work_dirs/   (gitignored)       # training checkpoints/logs from mmdet3d
  logs/        (gitignored)       # convert_all.sh run logs
```

`data/`, `work_dirs/`, and `logs/` are all gitignored on purpose — they're large and
fully regenerable from the raw LAS files + HD map SHP, which themselves aren't
committed either. When moving to a new machine, copy over (rsync/scp) just the raw
inputs, not these generated directories.

## 4. Running it

Inputs needed: raw LAS files (one per corridor segment) and an HD map SHP export
directory containing `B2_SURFACELINEMARK.shp` (lane lines) and `B3_SURFACEMARK.shp`
(crosswalks).

```bash
# 1. convert every *.las in a directory into training tiles
./training/convert_all.sh <las_dir> <hdmap_dir>          # defaults: ~/다운로드, ~/ayg-dna-pcn

# 2. build train/val split (whole source files held out for val, to avoid
#    leaking near-identical tiles between train/val)
python3 training/make_info_pkl.py training/data/lane3d --val-source-ids 19-002,18-003

# 3. train (from the repo root)
mim train mmdet3d training/configs/cylinder3d_lane3d.py

# 4. run inference on one source file's tiles
python3 training/run_inference.py 19-002 out.geojson \
    --checkpoint training/work_dirs/cylinder3d_lane3d_full/epoch_30.pth \
    --data-root training/data/lane3d

# 5. turn predictions into a Shapefile (loadable via the sidebar's "Import SHP overlay")
python3 training/vectorize_predictions.py out.geojson out.shp --epsg EPSG:32652
```
