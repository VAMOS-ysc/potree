// Converts ml/vectorize_predictions.py output (LineString = line_marking,
// Polygon = crosswalk) into native Potree Measure objects - the same class the
// "Lane"/"Area" toolbar buttons create (see src/viewer/sidebar.js) - instead of
// a read-only overlay. That makes predictions show up in the Scene panel where
// markers can be dragged/deleted and lane type/color retagged, then exported
// through the existing "SHP" button (src/exporter/GeoJSONExporter.js writes a
// Kind/Type field for isLane/isCrosswalk measurements so the export can be fed
// straight back into ml/rasterize.py --lines/--crosswalks for the next training
// round).
import * as THREE from "../../libs/three.js/build/three.module.js";

export function importPredictionsAsMeasurements(viewer, geojson) {
	const counts = {lane: 0, crosswalk: 0};

	for (const f of geojson.features) {
		const cls = f.properties.class;

		if (cls === "lane" && f.geometry.type === "LineString") {
			const measure = new Potree.Measure();
			measure.name = "Lane (predicted)";
			measure.showDistances = true;
			measure.showArea = false;
			measure.closed = false;
			measure.maxMarkers = Infinity;
			measure.isLane = true;
			measure.laneType = "solid";
			measure.laneColor = "white";
			measure.color = new THREE.Color(0xffffff);

			for (const c of f.geometry.coordinates) {
				measure.addMarker(new THREE.Vector3(c[0], c[1], c[2]));
			}
			viewer.scene.addMeasurement(measure);
			counts.lane++;
		} else if (cls === "crosswalk" && f.geometry.type === "Polygon") {
			const measure = new Potree.Measure();
			measure.name = "Crosswalk (predicted)";
			measure.showDistances = false;
			measure.showArea = true;
			measure.closed = true;
			measure.maxMarkers = Infinity;
			measure.isCrosswalk = true;
			measure.color = new THREE.Color(0x2090ff);

			// GeoJSON polygon rings repeat the first point as the last one to
			// close the loop - Measure's own `closed` flag already connects the
			// last marker back to the first, so drop the duplicate or you get a
			// degenerate zero-length final edge.
			const [exterior] = f.geometry.coordinates;
			for (const c of exterior.slice(0, -1)) {
				measure.addMarker(new THREE.Vector3(c[0], c[1], c[2]));
			}
			viewer.scene.addMeasurement(measure);
			counts.crosswalk++;
		}
	}

	return counts;
}
