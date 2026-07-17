"""
One-stop tool: upload a LAS/LAZ file, auto-process it (ground classification +
PotreeConverter), and serve it through the potree viewer for lane digitizing.
Also exposes a GeoJSON -> Shapefile export endpoint.

Run with:
    python webapp/server.py
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
POINTCLOUDS_DIR = REPO_ROOT / "pointclouds"
ML_DIR = REPO_ROOT / "ml"
POTREE_CONVERTER = REPO_ROOT / "PotreeConverter" / "PotreeConverter"

app = FastAPI()


def find_model_dir() -> Path | None:
	"""Picks the checkpoint dir to use for /api/predict: the highest-numbered
	ml/checkpoints* directory (ml/checkpoints, ml/checkpoints_v2, ..._v6, ...)
	that actually has both best.pt and config.json - so a new train.py run just
	needs the usual checkpoints_vN naming to become the one the webapp serves,
	no code change here.

	Deliberately NOT based on each run's recorded history.json val_mIoU - tried
	that (2026-07-16) and it picked an old run whose mIoU looked better only
	because it validated on a single, easier scene; mIoU isn't comparable
	across runs with different --val-scenes. A higher version number is not a
	guaranteed improvement either (confirmed same day: v7 scored worse than v6
	on every class despite more Dice weight/capacity) - it's on whoever trains
	a new version to confirm it's actually better (e.g. compare per-class IoU
	against the previous best) before that version number is used, same as any
	other model rollout. A checkpoints_vN directory that turned out worse
	should be renamed so it's excluded from this glob.
	"""
	candidates = []
	for d in ML_DIR.glob("checkpoints*"):
		if not (d / "best.pt").exists() or not (d / "config.json").exists():
			continue
		m = re.fullmatch(r"checkpoints(?:_v(\d+))?", d.name)
		if m:
			candidates.append((int(m.group(1) or 1), d))
	if not candidates:
		return None
	return max(candidates, key=lambda t: t[0])[1]


def run_ml_script(script: str, args: list[str]):
	"""Runs one of the ml/ scripts with the same Python interpreter running this
	server - per README, ml/requirements.txt and webapp/requirements.txt are
	meant to be installed into one shared env, so sys.executable already has
	torch/rasterio/etc. if that setup was followed (no hardcoded env/paths)."""
	cmd = [sys.executable, str(ML_DIR / script), *args]
	result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ML_DIR))
	if result.returncode != 0:
		raise HTTPException(500, f"{script} failed: {result.stderr[-4000:]}")
	return result


def sanitize_name(filename: str) -> str:
	stem = Path(filename).stem
	stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "dataset"
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return f"{stem}_{timestamp}"


def get_proj4(las_path: Path, epsg_override: str | None = None) -> tuple[str, str]:
	# Some LAS exports (seen with a handful of this project's scans) carry no SRS
	# in the header at all - pdal then returns an empty proj4/wkt. There's no way
	# to recover the real CRS from the file itself in that case, so the frontend
	# asks the user for an EPSG code and retries with epsg_override set.
	if epsg_override:
		result = subprocess.run(["gdalsrsinfo", "-o", "proj4", epsg_override], capture_output=True, text=True)
		proj4 = result.stdout.strip().strip("'\"")
		if result.returncode != 0 or not proj4:
			raise HTTPException(400, f"'{epsg_override}' is not a recognized EPSG code.")
		return epsg_override, proj4

	result = subprocess.run(
		["pdal", "info", "--metadata", str(las_path)],
		capture_output=True, text=True, check=True,
	)
	metadata = json.loads(result.stdout)["metadata"]
	srs = metadata.get("srs", {})
	proj4 = srs.get("proj4")
	epsg = None
	for component in srs.get("json", {}).get("components", [srs.get("json", {})]):
		code = component.get("id", {}).get("code")
		if code:
			epsg = f"EPSG:{code}"
			break
	if not proj4:
		raise HTTPException(422, "Could not determine the coordinate system (SRS) of the uploaded file - "
			"it has no embedded SRS. Provide one manually (e.g. EPSG:32652).")
	return epsg or "EPSG:UNKNOWN", proj4


def is_already_classified(las_path: Path) -> bool:
	result = subprocess.run(
		["pdal", "info", "--stats", "--dimensions", "Classification", str(las_path)],
		capture_output=True, text=True, check=True,
	)
	stats = json.loads(result.stdout)["stats"]["statistic"][0]
	return stats["maximum"] > 0


def pad_bounding_box(las_path: Path, pad: float = 0.01):
	"""Widens the LAS/LAZ header's bounding box by `pad` meters on every side, in place.

	Some files (seen after PDAL's SMRF+writers.las round-trip, but also possible in
	source files) end up with a header bounding box that's a few floating-point ULPs
	too tight for the actual quantized point coordinates - e.g. a point re-materializes
	as 4141384.7800000003 while the header max_y says 4141384.78. PotreeConverter's
	LASzip-based chunker treats that as fatal ("encountered point outside bounding
	box") even though the file is otherwise valid. Padding by 1cm is far below any
	meaningful precision for lidar data and sidesteps the strict check entirely.
	Offsets 179-227 (6 doubles: max/min X, max/min Y, max/min Z) are part of the
	public header block shared by every LAS/LAZ version (1.0-1.4), so this is safe
	regardless of the file's version.
	"""
	with open(las_path, "r+b") as f:
		f.seek(179)
		maxx, minx, maxy, miny, maxz, minz = struct.unpack("<6d", f.read(48))

		f.seek(179)
		f.write(struct.pack("<6d", maxx + pad, minx - pad, maxy + pad, miny - pad, maxz + pad, minz - pad))


def run_ground_classification(las_path: Path, out_path: Path):
	pipeline = {
		"pipeline": [
			{"type": "readers.las", "filename": str(las_path)},
			{"type": "filters.smrf"},
			{"type": "writers.las", "filename": str(out_path), "minor_version": 2, "dataformat_id": 3},
		]
	}
	with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
		json.dump(pipeline, f)
		pipeline_path = f.name
	subprocess.run(["pdal", "pipeline", pipeline_path], check=True, capture_output=True, text=True)


@app.post("/api/process")
async def process_pointcloud(file: UploadFile = File(...), epsg: str | None = None):
	suffix = Path(file.filename).suffix.lower()
	if suffix not in (".las", ".laz"):
		raise HTTPException(400, "Only .las/.laz files are supported. PCD is not supported yet "
			"(ego-frame PCD data lacks absolute coordinates without vehicle pose).")

	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp)
		uploaded_path = tmp_path / file.filename
		with open(uploaded_path, "wb") as f:
			shutil.copyfileobj(file.file, f)

		try:
			epsg, proj4 = get_proj4(uploaded_path, epsg_override=epsg)
		except HTTPException:
			raise
		except subprocess.CalledProcessError as e:
			raise HTTPException(400, f"Failed to read file metadata: {e.stderr}")

		if is_already_classified(uploaded_path):
			source_for_conversion = uploaded_path
		else:
			classified_path = tmp_path / "classified.las"
			try:
				run_ground_classification(uploaded_path, classified_path)
			except subprocess.CalledProcessError as e:
				raise HTTPException(500, f"Ground classification failed: {e.stderr}")
			source_for_conversion = classified_path

		pad_bounding_box(source_for_conversion)

		dataset_id = sanitize_name(file.filename)
		out_dir = POINTCLOUDS_DIR / dataset_id
		out_dir.mkdir(parents=True, exist_ok=True)

		# The bundled binary needs its bundled liblaszip.so (unversioned name), which isn't on the
		# system loader path by default even when a system liblaszip is installed under a versioned name.
		env = {**os.environ, "LD_LIBRARY_PATH": str(POTREE_CONVERTER.parent)}
		result = subprocess.run(
			[str(POTREE_CONVERTER), str(source_for_conversion), "-o", str(out_dir), "--projection", epsg],
			capture_output=True, text=True, env=env,
		)
		if result.returncode != 0:
			shutil.rmtree(out_dir, ignore_errors=True)
			raise HTTPException(500, f"PotreeConverter failed: {result.stderr}")

		meta = {"epsg": epsg, "proj4": proj4, "sourceFile": file.filename}
		(out_dir / "potree_meta.json").write_text(json.dumps(meta, indent=2))

		# Kept around so /api/predict can rasterize intensity/height from the same
		# LAS later on demand, without re-uploading - source_for_conversion (not
		# uploaded_path) so it's already ground-classified if it needed to be.
		shutil.copy(source_for_conversion, out_dir / "source.las")

	return JSONResponse({"dataset": dataset_id})


@app.post("/api/predict")
async def predict(dataset: str):
	out_dir = (POINTCLOUDS_DIR / dataset).resolve()
	if out_dir.parent != POINTCLOUDS_DIR.resolve():
		raise HTTPException(400, "Invalid dataset id")
	source_las = out_dir / "source.las"
	if not source_las.exists():
		raise HTTPException(404, f"No source LAS stored for dataset '{dataset}' "
			"(uploaded before the predict feature was added - re-upload it).")

	model_dir = find_model_dir()
	if model_dir is None:
		raise HTTPException(500, "No trained checkpoint found under ml/checkpoints* "
			"(need a best.pt + config.json pair - run ml/train.py first).")

	scene_dir = out_dir / "ml_scene"
	if not (scene_dir / "intensity.tif").exists():
		run_ml_script("rasterize.py", ["--las", str(source_las), "--out", str(scene_dir)])

	mask_path = scene_dir / "pred_mask.tif"
	run_ml_script("infer.py", [
		"--scene-dir", str(scene_dir),
		"--checkpoint", str(model_dir / "best.pt"),
		"--config", str(model_dir / "config.json"),
		"--out", str(out_dir / "predictions_points.geojson"),
		"--out-mask", str(mask_path),
	])

	vectors_path = out_dir / "predictions_vectors.geojson"
	run_ml_script("vectorize_predictions.py", [
		"--scene-dir", str(scene_dir),
		"--mask", str(mask_path),
		"--out", str(vectors_path),
	])

	return JSONResponse({"vectors": f"/pointclouds/{dataset}/predictions_vectors.geojson",
	                      "model": model_dir.name})



# A single .shp can only hold one geometry type - ogr2ogr picks the first
# feature's type for the layer and drops the rest, silently losing data. The
# scene mixes LineString (lanes), Polygon (crosswalks/areas) and Point (edge
# length / area labels), so features are split by geometry type into their own
# FeatureCollection/shp before conversion. "lines"/"polygons" match the names
# ml/rasterize.py's --lines/--crosswalks expect, so this zip's contents can be
# fed straight back in to regenerate training masks from a corrected prediction.
GEOJSON_TYPE_TO_SHP_NAME = {"LineString": "lines", "Polygon": "polygons", "Point": "points"}


@app.post("/api/export-shp")
async def export_shp(request: Request, epsg: str | None = None):
	geojson_text = (await request.body()).decode("utf-8")
	geojson = json.loads(geojson_text)

	by_type: dict[str, list] = {}
	for feature in geojson.get("features", []):
		geom_type = feature.get("geometry", {}).get("type")
		by_type.setdefault(geom_type, []).append(feature)

	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp)
		wrote_any = False

		for geom_type, features in by_type.items():
			shp_name = GEOJSON_TYPE_TO_SHP_NAME.get(geom_type)
			if not shp_name:
				continue

			geojson_path = tmp_path / f"{shp_name}.geojson"
			geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

			shp_path = tmp_path / f"{shp_name}.shp"
			# The GeoJSON coordinates are raw scene coordinates (already in the
			# pointcloud's real-world CRS, e.g. UTM meters), not WGS84 lon/lat as
			# GeoJSON normally implies. -a_srs assigns the correct CRS without
			# reprojecting the (already-correct) values.
			cmd = ["ogr2ogr", "-f", "ESRI Shapefile"]
			if epsg:
				cmd += ["-a_srs", epsg]
			cmd += [str(shp_path), str(geojson_path)]
			result = subprocess.run(cmd, capture_output=True, text=True)
			if result.returncode != 0:
				raise HTTPException(500, f"SHP conversion failed ({shp_name}): {result.stderr}")
			wrote_any = True

		if not wrote_any:
			raise HTTPException(400, "No exportable geometry (Point/LineString/Polygon) in the given measurements.")

		zip_path = tmp_path / "measure_shp.zip"
		with zipfile.ZipFile(zip_path, "w") as zf:
			for component in tmp_path.iterdir():
				if component.suffix != ".geojson" and component.name != zip_path.name:
					zf.write(component, component.name)

		final_zip = Path(tempfile.mkstemp(suffix=".zip")[1])
		shutil.copy(zip_path, final_zip)

	return FileResponse(final_zip, media_type="application/zip", filename="measure_shp.zip")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
	return (REPO_ROOT / "webapp" / "upload.html").read_text()


# Must be registered last: this mount is a catch-all for everything else
# (examples/, build/, libs/, resources/, pointclouds/).
app.mount("/", StaticFiles(directory=str(REPO_ROOT), html=True), name="static")


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8080)
