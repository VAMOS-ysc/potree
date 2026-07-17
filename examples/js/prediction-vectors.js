// Draws U-Net lane/crosswalk predictions (ml/vectorize_predictions.py output:
// LineString for line_marking, Polygon for crosswalk) onto a Potree scene.
// Shared by test_predictions.html and lane_digitize.html so the two don't drift.
import * as THREE from "../../libs/three.js/build/three.module.js";

// Lifted a few cm above the recorded height so the overlay doesn't z-fight
// with the real ground points sitting at that same spot.
const Z_LIFT = 0.08;
const COLORS = {lane: 0xff2020, crosswalk: 0x2090ff};

// raw UTM coordinates (~3.2e5, 4.1e6) blow float32 precision once they hit the
// model-view-projection matrix chain in the vertex shader - points render
// unstably or vanish. Fix: store small LOCAL vertex values (offset subtracted
// out) and carry the large offset on the object's own .position instead,
// exactly like the octree loader already does for the point cloud itself.
export function renderPredictionVectors(viewer, pointcloud, geojson) {
	const offset = pointcloud.position;
	const counts = {lane: 0, crosswalk: 0};
	const group = new THREE.Group();

	for (const f of geojson.features) {
		const cls = f.properties.class;
		const color = COLORS[cls] ?? 0xffffff;
		counts[cls] = (counts[cls] ?? 0) + 1;

		if (f.geometry.type === "LineString") {
			const coords = f.geometry.coordinates;
			const positions = new Float32Array(coords.length * 3);
			for (let i = 0; i < coords.length; i++) {
				positions[i * 3 + 0] = coords[i][0] - offset.x;
				positions[i * 3 + 1] = coords[i][1] - offset.y;
				positions[i * 3 + 2] = coords[i][2] - offset.z + Z_LIFT;
			}
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
			const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({color, linewidth: 2, depthTest: true, depthWrite: true}));
			line.position.copy(offset);
			line.renderOrder = 10;
			group.add(line);
		} else if (f.geometry.type === "Polygon") {
			// crosswalks read as locally-flat pavement - fill at the ring's mean Z
			// (THREE.ShapeGeometry only supports flat 2D shapes) and draw the true
			// per-vertex-Z ring on top as an outline so any real undulation is
			// still visible there.
			const [exterior, ...holes] = f.geometry.coordinates;
			const meanZ = exterior.reduce((s, c) => s + c[2], 0) / exterior.length;

			const shape = new THREE.Shape(exterior.map(c => new THREE.Vector2(c[0] - offset.x, c[1] - offset.y)));
			for (const hole of holes) {
				shape.holes.push(new THREE.Path(hole.map(c => new THREE.Vector2(c[0] - offset.x, c[1] - offset.y))));
			}
			const fillGeom = new THREE.ShapeGeometry(shape);
			const fill = new THREE.Mesh(fillGeom, new THREE.MeshBasicMaterial({
				color, transparent: true, opacity: 0.35, side: THREE.DoubleSide,
				depthTest: true, depthWrite: true,
			}));
			// meanZ is already an absolute world Z (from the geojson's real
			// elevation values) - unlike the outline/points code above, this
			// mesh's local vertices were built directly in world XY (only
			// offset.x/y subtracted, no Z touched), so position.z must be meanZ
			// alone. Adding offset.z here double-counts the point cloud's own
			// Z-offset and floats the fill high above the ground (confirmed
			// visually - outline sat correctly on the point cloud, fill floated
			// dozens of meters above it).
			fill.position.set(offset.x, offset.y, meanZ + Z_LIFT);
			fill.renderOrder = 9;
			group.add(fill);

			for (const ring of [exterior, ...holes]) {
				const positions = new Float32Array(ring.length * 3);
				for (let i = 0; i < ring.length; i++) {
					positions[i * 3 + 0] = ring[i][0] - offset.x;
					positions[i * 3 + 1] = ring[i][1] - offset.y;
					positions[i * 3 + 2] = ring[i][2] - offset.z + Z_LIFT;
				}
				const geometry = new THREE.BufferGeometry();
				geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
				const loop = new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({color, depthTest: true, depthWrite: true}));
				loop.position.copy(offset);
				loop.renderOrder = 10;
				group.add(loop);
			}
		}
	}

	viewer.scene.scene.add(group);
	return {group, counts};
}
