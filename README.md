
# Lane Digitize Tool

A desktop tool for extracting road lane markings from mobile LiDAR scans (LAS/LAZ) and
exporting them as ESRI Shapefiles. Built on top of [Potree](#about) (WebGL point cloud
viewer), [PotreeConverter](https://github.com/potree/PotreeConverter), [PDAL](https://pdal.io/)
and [GDAL](https://gdal.org/).

Upload a LAS/LAZ file → it's automatically ground-classified (PDAL `filters.smrf`) and
converted to a streamable point cloud (PotreeConverter) → digitize lane lines directly on
the point cloud with the "Lane" tool → export as a georeferenced Shapefile.

## Prerequisites

* [Node.js](https://nodejs.org/) + npm
* Python 3.9+ with:
  * `pip install -r webapp/requirements.txt`
  * [PDAL](https://pdal.io/) (e.g. `conda install -c conda-forge pdal` or your distro's package manager) — used for ground classification and reading LAS coordinate system metadata
  * [GDAL](https://gdal.org/) CLI tools (`ogr2ogr`, `gdal_rasterize`, `gdalsrsinfo`, `gdal_create`) on PATH — used for the Shapefile export and the `ml/` data prep scripts
* Linux x86_64: the bundled `PotreeConverter/PotreeConverter` binary works out of the box.
  Windows/Mac: replace it with a build from [potree/PotreeConverter](https://github.com/potree/PotreeConverter).

## Setup & Run

```bash
npm install        # installs gulp/rollup build tooling + electron, and builds potree (postinstall)
npm run desktop     # starts the backend and opens the app window
```

The app opens on an empty viewer. Use the Scene panel's Import button to pick a `.las`/`.laz`
file, wait for processing, then use the "Lane" toolbar icon to click along lane markings.
Select a drawn lane in the Scene panel to tag its type (solid/dashed/stop line) and color.
Use the "SHP" button in the Scene panel's export row to download a Shapefile.

If you'd rather run it as a plain web app instead of the Electron window, `python
webapp/server.py` starts just the backend; visit `http://localhost:8080/upload`.

## Lane/Crosswalk Segmentation Data Prep (`ml/`)

`ml/rasterize.py` converts a LAS/LAZ scan plus 정밀도로지도-format Shapefiles
(`B2_SURFACELINEMARK` for lane/stop lines, `B3_SURFACEMARK` for crosswalks) into
pixel-aligned GeoTIFF rasters — an (intensity + height) input image and a class-id
mask — for training a segmentation model (e.g. U-Net). It's a standalone
preprocessing step; it doesn't touch the viewer or webapp.

```bash
pip install -r ml/requirements.txt   # PDAL/GDAL CLI tools are shared with the webapp, see Prerequisites above

python ml/rasterize.py \
  --las path/to/scan.las \
  --lines path/to/B2_SURFACELINEMARK.shp \
  --crosswalks path/to/B3_SURFACEMARK.shp \
  --out ml/out/scan \
  --resolution 0.05 \
  --preview
```

Outputs `intensity.tif`, `height.tif`, `mask.tif` (0=background, 1=lane_line,
2=stop_line, 3=crosswalk, 4=other — see `CLASS_NAMES` in the script), and
`meta.json` recording the bounds/resolution/EPSG/class mapping used. The
Kind→class mapping is based on the 정밀도로지도 제작 매뉴얼(2020.12) 9.3.7–9.3.8
code tables and can be overridden with `--class-map some.json`.

Any of `--las`, `--lines`, `--crosswalks` may be omitted (e.g. rasterize just a
mask from Shapefiles, or just an intensity/height raster from a LAS file); the
output region defaults to the union of whichever inputs are given, or override
with `--bounds XMIN YMIN XMAX YMAX`.

If the Shapefile's CRS differs from the LAS (common - 정밀도로지도 deliverables are
often lon/lat EPSG:4326, sometimes without even a `.prj` to say so), it's
reprojected to the LAS's CRS automatically; see `--epsg`/`--shp-srs` for the SHP
layers that have no `.prj` at all.

### Training (`ml/train.py`)

Trains a U-Net on however many `ml/rasterize.py` output directories ("scenes")
you point it at, patch-tiling each one and splitting train/val **by scene** (not
by patch) so validation isn't leaking adjacent-patch context from the same road.

```bash
# rasterize a few scenes first, e.g.:
#   python ml/rasterize.py --las d1.las --lines B2....shp --crosswalks B3....shp --out ml/out/scene1 --resolution 0.05
#   python ml/rasterize.py --las d2.las --lines B2....shp --crosswalks B3....shp --out ml/out/scene2 --resolution 0.05
#   ...

python ml/train.py \
  --data-root ml/out \
  --val-scenes scene2 \
  --patch-size 256 --batch-size 8 --epochs 30 \
  --out ml/checkpoints
```

Handles the severe class imbalance (background is 98%+ of pixels) with inverse-frequency
class weights + Dice loss, and excludes pixels with no LiDAR return at all from the loss
(rather than training them as "background"). Saves `last.pt`/`best.pt` (by val mIoU),
`history.json`, and `config.json` to `--out`. With only one scene rasterized, it falls
back to a random patch-level split with a warning instead of erroring — fine for a
smoke test, not for a real accuracy number.

GPU note: if you're power-limited (e.g. a PSU that can't sustain a GPU's full TDP under
sustained training load), cap it at the driver level once (`sudo nvidia-smi -pl <watts>`,
made to persist across reboots with a systemd unit) rather than in this script - the
driver clocks down to stay under the limit regardless of batch size.

---

# About

* Potree is a free open-source WebGL based point cloud renderer for large point clouds. It is based on the [TU Wien Scanopy project](https://www.cg.tuwien.ac.at/research/projects/Scanopy/) and research projects [Harvest4D](https://harvest4d.org/), [GCD Doctoral College](https://gcd.tuwien.ac.at/) and [Superhumans](https://www.cg.tuwien.ac.at/research/projects/Superhumans/).
* Newest information and work in progress is usually available on [twitter](https://twitter.com/m_schuetz)
* Contact: Markus Schütz (mschuetz@potree.org)
* References: 
    * [Potree: Rendering Large Point Clouds in Web Browsers](https://www.cg.tuwien.ac.at/research/publications/2016/SCHUETZ-2016-POT/SCHUETZ-2016-POT-thesis.pdf) (2016)
    * [Fast Out-of-Core Octree Generation for Massive Point Clouds](https://www.cg.tuwien.ac.at/research/publications/2020/SCHUETZ-2020-MPC/) (2020)
    
<a href="http://potree.org/wp/demo/" target="_blank"> ![](./docs/images/potree_screens.png) </a>

# Getting Started

### Install on your PC

Install [node.js](http://nodejs.org/)

Install dependencies, as specified in package.json, and create a build in ./build/potree.

```bash
npm install
```

### Run on your PC

Use the `npm start` command to 

* create ./build/potree 
* watch for changes to the source code and automatically create a new build on change
* start a web server at localhost:1234. 

Go to http://localhost:1234/examples/ to test the examples.

### Deploy to a server

* Simply upload the Potree folderm with all your point clouds, the build directory, and your html files to a web server.
* It is not required to install node.js on your webserver. All you need is to host your files online. 

### Convert Point Clouds to Potree Format

Download [PotreeConverter](https://github.com/potree/PotreeConverter) and run it like this:

    ./PotreeConverter.exe C:/pointclouds/data.las -o C:/pointclouds/data_converted

Copy the converted directory into &lt;potreeDirectory&gt;/pointclouds/data_converted. Then, duplicate and rename one of the examples and modify the path in the html file to your own point cloud.

# Downloads

* [Potree](https://github.com/potree/potree/releases)
* [PotreeConverter ](https://github.com/potree/PotreeConverter/releases) - Convert your point cloud to the Potree format.
* [PotreeDesktop ](https://github.com/potree/PotreeDesktop/releases) - Desktop version of Potree. Allows drag&drop of point clouds into the viewer.

# Examples

<table>
	<tr>
		<td style="padding: 0px">
			<a href="http://potree.org/potree/examples/viewer.html" target="_blank">
				<img src="examples/thumbnails/viewer.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/ca13.html" target="_blank">
				<img src="examples/thumbnails/ca13.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/cesium_retz.html" target="_blank">
				<img src="examples/thumbnails/cesium_retz.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/classifications.html" target="_blank">
				<img src="examples/thumbnails/classifications.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/features_sorvilier.html" target="_blank">
				<img src="examples/thumbnails/features_sorvilier.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/toolbar.html" target="_blank">
				<img src="examples/thumbnails/toolbar.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Basic Viewer</th><th>CA13 (18 billion Points)</th><th>Retz (Potree + Cesium)</th><th>Classifications</th><th>Various Features</th><th>Toolbar</th>
	</tr>
</table>

<details>
<summary>More Examples</summary>


<table>
	<tr>
		<td>
			<a href="http://potree.org/potree/examples/load_project.html" target="_blank">
				<img src="examples/thumbnails/load_project.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/matcap.html" target="_blank">
				<img src="examples/thumbnails/matcap.jpg" width="100%" />
			</a>
		</td><td>
			<a href="https://potree.org/potree/examples/vr_heidentor.html" target="_blank">
				<img src="examples/thumbnails/heidentor.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/heidentor.html" target="_blank">
				<img src="examples/thumbnails/heidentor.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/lion.html" target="_blank">
				<img src="examples/thumbnails/lion.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/lion_las.html" target="_blank">
				<img src="examples/thumbnails/lion_las.png" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Load Project</th><th>Matcap</th><th>Virtual Reality</th><th>Heidentor</th><th>Lion</th><th>Lion LAS</th>
	</tr><tr>
		<td>
			<a href="http://potree.org/potree/examples/lion_laz.html" target="_blank">
				<img src="examples/thumbnails/lion_las.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/ept.html" target="_blank">
				<img src="examples/thumbnails/lion.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/ept_binary.html" target="_blank">
				<img src="examples/thumbnails/lion_las.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/ept_zstandard.html" target="_blank">
				<img src="examples/thumbnails/lion_las.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/clipping_volume.html" target="_blank">
				<img src="examples/thumbnails/clipping_volume.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/oriented_images.html" target="_blank">
				<img src="examples/thumbnails/oriented_images.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Lion LAZ</th><th>EPT</th><th>EPT Binary</th><th>EPT zstandard</th><th>Clipping Volume</th><th>Oriented Images</th>
	</tr><tr>
		<td>
			<a href="http://potree.org/potree/examples/elevation_profile.html" target="_blank">
				<img src="examples/thumbnails/elevation_profile.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/measurements.html" target="_blank">
				<img src="examples/thumbnails/measurements.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/meshes.html" target="_blank">
				<img src="examples/thumbnails/meshes.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/multiple_pointclouds.html" target="_blank">
				<img src="examples/thumbnails/multiple_point_clouds.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/camera_animation.html" target="_blank">
				<img src="examples/thumbnails/camera_animation.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/features_ca13.html" target="_blank">
				<img src="examples/thumbnails/features_ca13.png" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Elevation Profile</th><th>Measurements</th><th>Meshes</th><th>Multiple Point Clouds</th><th>Camera Animation</th><th>Features (CA13)</th>
	</tr><tr>
		<td>
			<a href="http://potree.org/potree/examples/annotations.html" target="_blank">
				<img src="examples/thumbnails/annotations.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/annotation_hierarchy.html" target="_blank">
				<img src="examples/thumbnails/annotation_hierarchy.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/animation_paths.html" target="_blank">
				<img src="examples/thumbnails/animation_paths.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/shapefiles.html" target="_blank">
				<img src="examples/thumbnails/shapefiles.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/cesium_ca13.html" target="_blank">
				<img src="examples/thumbnails/cesium_ca13.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/geopackage.html" target="_blank">
				<img src="examples/thumbnails/geopackage.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Annotations</th><th>Hierarchical Annotations</th><th>Animation Path</th><th>Shapefiles</th><th>Cesium CA13</th><th>Geopackage</th>
	</tr><tr>
		<td>
			<a href="http://potree.org/potree/examples/cesium_sorvilier.html" target="_blank">
				<img src="examples/thumbnails/cesium_sorvilier.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/custom_sidebar_section.html" target="_blank">
				<img src="examples/thumbnails/custom_sidebar_section.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/embedded_iframe.html" target="_blank">
				<img src="examples/thumbnails/embedded_iframe.png" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/gradient_colors.html" target="_blank">
				<img src="examples/thumbnails/gradient_colors.png" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Cesium Sorvilier</th><th>Custom Sidebar Section</th><th>Embedded Iframe</th><th>Gradient Colors</th>
	</tr>
</table>
</details>

# VR

<table>
	<tr>
		<td>
			<a href="https://potree.org/potree/examples/vr_heidentor.html" target="_blank">
				<img src="examples/thumbnails/heidentor.jpg" width="100%" />
			</a>
		</td><td>
			<a href="https://potree.org/potree/examples/vr_eclepens.html" target="_blank">
				<img src="examples/thumbnails/eclepens.jpg" width="100%" />
			</a>
		</td><td>
			<a href="https://potree.org/potree/examples/vr_morro_bay.html" target="_blank">
				<img src="examples/thumbnails/ca13.png" width="100%" />
			</a>
		</td><td>
			<a href="https://potree.org/potree/examples/vr_lion.html" target="_blank">
				<img src="examples/thumbnails/lion.png" width="100%" />
			</a>
		</td><td>
			<a href="https://potree.org/potree/examples/vr_dechen_cave.html" target="_blank">
				<img src="examples/thumbnails/dechen_cave.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Heidentor</th><th>Eclepens</th><th>Morro Bay</th><th>Lion</th><th>Dechen Cave</th>
	</tr>
</table>

# Showcase

<table>
	<tr>
		<td>
			<a href="http://potree.org/potree/examples/showcase/matterhorn.html" target="_blank">
				<img src="examples/thumbnails/matterhorn.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/retz.html" target="_blank">
				<img src="examples/thumbnails/retz.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/lake_tahoe.html" target="_blank">
				<img src="examples/thumbnails/lake_tahoe.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/sorvilier.html" target="_blank">
				<img src="examples/thumbnails/vol_total.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/grab_15.html" target="_blank">
				<img src="examples/thumbnails/grab_15.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/tern_auscover_chowilla.html" target="_blank">
				<img src="examples/thumbnails/chowilla.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Matterhorn</th><th>Retz</th><th>Lake Tahoe</th><th>Sorvilier</th><th>Grave</th><th>Chowilla</th>
	</tr>
</table>

<details>
<summary>More</summary>

<table>
	<tr>
		<td>
			<a href="http://potree.org/potree/examples/showcase/chiller.html" target="_blank">
				<img src="examples/thumbnails/chiller.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/cooler_tower.html" target="_blank">
				<img src="examples/thumbnails/cooler_tower.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/dechen_cave.html" target="_blank">
				<img src="examples/thumbnails/dechen_cave.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/doverMillRuins.html" target="_blank">
				<img src="examples/thumbnails/DoverMillRuins.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/eclepens.html" target="_blank">
				<img src="examples/thumbnails/eclepens.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/heidentor.html" target="_blank">
				<img src="examples/thumbnails/heidentor.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Chiller</th><th>Cooler</th><th>Dechen Cave</th><th>Ruins</th><th>Eclepens</th><th>Heidentor</th>
	</tr><tr>
		<td>
			<a href="http://potree.org/potree/examples/showcase/land_building.html" target="_blank">
				<img src="examples/thumbnails/land_building.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/LDHI_module.html" target="_blank">
				<img src="examples/thumbnails/LDHI_module.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/lion_head_simone_garagnani.html" target="_blank">
				<img src="examples/thumbnails/lion_head.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/overpass.html" target="_blank">
				<img src="examples/thumbnails/overpass.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/pielach.html" target="_blank">
				<img src="examples/thumbnails/pielach.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/pompei.html" target="_blank">
				<img src="examples/thumbnails/pompei.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Building</th><th>LDHI</th><th>Lion Head</th><th>Overpass</th><th>Pielach</th><th>pompei</th>
	</tr><tr>
		<td>
			<a href="http://potree.org/potree/examples/showcase/santorini.html" target="_blank">
				<img src="examples/thumbnails/santorini.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/skatepark.html" target="_blank">
				<img src="examples/thumbnails/skatepark.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/subsea_equipment.html" target="_blank">
				<img src="examples/thumbnails/subsea_equipment.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/subsea_manifold.html" target="_blank">
				<img src="examples/thumbnails/subseamanifold.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/westend_palais.html" target="_blank">
				<img src="examples/thumbnails/westend_palais.jpg" width="100%" />
			</a>
		</td><td>
			<a href="http://potree.org/potree/examples/showcase/whitby.html" target="_blank">
				<img src="examples/thumbnails/whitby.jpg" width="100%" />
			</a>
		</td>
	</tr>
	<tr>
		<th>Santorini</th><th>Skatepark</th><th>Subsea Eq.</th><th>Subsea Man.</th><th>Westend Palais</th><th>Whitby</th>
	</tr>
</table>

</details>

# Funding

Potree is funded by a combination of research projects, companies and institutions. 

Research projects who's funding contributes to Potree:

<table>
	<tr>
		<th>Project Name</th>
		<th>Funding Agency</th>
	</tr>
	<tr>
		<td><a href="https://www.netidee.at/potree-next">Potree-Next</a></td>
		<td><a href="https://www.netidee.at/">Netidee</a></td>
	</tr>
	<tr>
		<td><a href="https://projekte.ffg.at/projekt/3851914">LargeClouds2BIM</a></td>
		<td><a href="https://www.ffg.at/">FFG</a></td>
	</tr>
	<tr>
		<td><a href="https://harvest4d.org/">Harvest4D</a></td>
		<td><a href="https://ec.europa.eu/transport/themes/research/fp7_en">EU 7th Framework Program 323567</a></td>
	</tr>
	<tr>
		<td><a href="https://gcd.tuwien.ac.at/">GCD Doctoral College</a></td>
		<td><a href="https://www.tuwien.at/en/">TU Wien</a></td>
	</tr>
	<tr>
		<td><a href="https://www.cg.tuwien.ac.at/research/projects/Superhumans/">Superhumans</a></td>
		<td><a href="https://www.fwf.ac.at/">FWF</a></td>
	</tr>
</table>

We would like to thank our sponsors for their financial contributions that keep this project up and running!

<table>
	<tr>
		<th>
			Diamond<br>
			€ 15,000+
		</th>
		<td>
			<a href="http://www.ne.ch/autorites/DDTE/SGRF/SITN/Pages/accueil.aspx">
				<img src="docs/sponsors/sitn_logo.png" height="80px"/> &nbsp;
			</a> &nbsp;
			<a href="http://www.synth3d.co">
				<img src="docs/sponsors/synth.png" height="120"/>
			</a> &nbsp;
			<a href="http://www.geocue.com">
				<img src="docs/sponsors/geocue.png" height="120px"/>
			</a> &nbsp;
			<a href="http://rapidlasso.com">
				<img src="./docs/sponsors/rapidlasso_square_256x2561.png" width="150" height="150"/>
			</a> &nbsp;
		</td>
	</tr>
	<tr>
		<th>
			Gold<br>
			€ 10,000+
		</th>
		<td>
			<a href="https://www.bart.gov">
				<img src="docs/sponsors/bart.png" height="100"/>
			</a>
		</td>
	</tr>
	<tr>
		<th>
			Silver<br>
			€ 5,000+
		</th>
		<td>
			<a href="https://www.skyebrowse.com/">
				<img src="docs/sponsors/SkyeBrowse.png" height="60"/> &nbsp;
			</a>
			<a href="https://biology.anu.edu.au/research/facilities/australian-plant-phenomics-facility-anu">
				<img src="docs/sponsors/APPF full logo.png" height="70"/> &nbsp;
			</a>
			<a href="https://www.limit-addict.fr/">
				<img src="docs/sponsors/limitaddict.png" height="45"/>
			</a>
			<a href="http://georepublic.info">
				<img src="docs/sponsors/georepublic.png" height="45"/>
			</a>
		</td>
	</tr>
	<tr>
		<th>
			Bronze<br>
			€ 1,000+
		</th>
		<td>
			<a href="https://www.kkc.co.jp/english/index.html/">
				<img src="docs/sponsors/kokusaikogyo_logo_02.jpg" height="40"/> &nbsp;
			</a>
			<a href="https://www.unstruk.com/">
				<img src="docs/sponsors/unstruk.png" height="33"/> &nbsp;
			</a>
			<a href="http://scanx.com/">
				<img src="docs/sponsors/scanx.jpg" height="33"/> &nbsp;
			</a>
			<a href="https://www.phoenixlidar.com/">
				<img src="docs/sponsors/PhoenixLidar_Logo.jpg" height="45"/> &nbsp;
			</a>
			<a href="https://www.eventart.at/">
				<img src="docs/sponsors/eventart.png" height="55"/> &nbsp;
			</a>
			<a href="https://www.geodelta.com/">
				<img src="docs/sponsors/geodelta.png" height="35"/> &nbsp;
			</a>
			<a href="https://www.e-cassini.fr/">
				<img src="docs/sponsors/e_cassini.jpg" height="70"/> &nbsp;
			</a>
			<a href="https://www.sogelink.fr/">
				<img src="docs/sponsors/SOGELINK_SO-EASY.png" height="40"/> &nbsp;
			</a>
			<b>Data-viewer</b>
			<a href="http://www.helimap.com/">
				<img src="docs/sponsors/helimap.gif" height="60"/> &nbsp;
			</a>
			<a href="http://www.vevey.ch/">
				<img src="docs/sponsors/vevey.png" height="60"/> &nbsp;
			</a>
			<a href="https://www.yverdon-les-bains.ch/">
				<img src="docs/sponsors/Logo-YLB.png" height="60"/> &nbsp;
			</a>
			<a href="http://archpro.lbg.ac.at">
				<img src="docs/sponsors/archpro_EN_small.png" height="60"/> 
			</a> &nbsp;
			<br>
			<a href="http://www.kts.co.jp">
				<img src="docs/sponsors/kts.png" height="32"/> &nbsp;
			</a>
			<a href="http://veesus.com">
				<img src="docs/sponsors/veesus_small.png" height="40"/> &nbsp;
			</a>
			<a href="http://www.sigeom.ch">
				<img src="docs/sponsors/logo_sigeom.png" height="40"/> &nbsp;
			</a>
		</td>
	</tr>
</table>



# Credits

* The multi-res-octree algorithms used by this viewer were developed at the Vienna University of Technology by Michael Wimmer and Claus Scheiblauer as part of the [Scanopy Project](http://www.cg.tuwien.ac.at/research/projects/Scanopy/).
* [Three.js](https://github.com/mrdoob/three.js), the WebGL 3D rendering library on which potree is built.
* [plas.io](http://plas.io/) point cloud viewer. LAS and LAZ support have been taken from the laslaz.js implementation of plas.io. Thanks to [Uday Verma](https://twitter.com/udaykverma) and [Howard Butler](https://twitter.com/howardbutler) for this!
* [Harvest4D](https://harvest4d.org/) Potree currently runs as Master Thesis under the Harvest4D Project
* Christian Boucheny (EDL developer) and Daniel Girardeau-Montaut ([CloudCompare](http://www.danielgm.net/cc/)). The EDL shader was adapted from the CloudCompare source code!
* [Martin Isenburg](http://rapidlasso.com/), [Georepublic](http://georepublic.de/en/),
[Veesus](http://veesus.com/), [Sigeom Sa](http://www.sigeom.ch/), [SITN](http://www.ne.ch/sitn), [LBI ArchPro](http://archpro.lbg.ac.at/),  [Pix4D](http://pix4d.com/) as well as all the contributers to potree and PotreeConverter and many more for their support.

# Bibtex 

```
@article{SCHUETZ-2020-MPC,
	title =      "Fast Out-of-Core Octree Generation for Massive Point Clouds",
	author =     "Markus Schütz and Stefan Ohrhallinger and Michael Wimmer",
	year =       "2020",
	month =      nov,
	journal =    "Computer Graphics Forum",
	volume =     "39",
	number =     "7",
	doi =        "10.1111/cgf.14134",
	pages =      "13",
	publisher =  "John Wiley & Sons, Inc.",
	pages =      "1--13",
	keywords =   "point clouds, point-based rendering, level of detail",
}
```
