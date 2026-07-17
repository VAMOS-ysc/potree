
import * as THREE from "../../libs/three.js/build/three.module.js";
import {TextSprite} from "../TextSprite.js";
import {Utils} from "../utils.js";
import {Line2} from "../../libs/three.js/lines/Line2.js";
import {LineGeometry} from "../../libs/three.js/lines/LineGeometry.js";
import {LineMaterial} from "../../libs/three.js/lines/LineMaterial.js";

function createHeightLine(){
	let lineGeometry = new LineGeometry();

	lineGeometry.setPositions([
		0, 0, 0,
		0, 0, 0,
	]);

	let lineMaterial = new LineMaterial({ 
		color: 0x00ff00, 
		dashSize: 5, 
		gapSize: 2,
		linewidth: 2, 
		resolution:  new THREE.Vector2(1000, 1000),
	});

	lineMaterial.depthTest = false;
	const heightEdge = new Line2(lineGeometry, lineMaterial);
	heightEdge.visible = false;

	//this.add(this.heightEdge);
	
	return heightEdge;
}

function createHeightLabel(){
	const heightLabel = new TextSprite('');

	heightLabel.setTextColor({r: 140, g: 250, b: 140, a: 1.0});
	heightLabel.setBorderColor({r: 0, g: 0, b: 0, a: 1.0});
	heightLabel.setBackgroundColor({r: 0, g: 0, b: 0, a: 1.0});
	heightLabel.fontsize = 16;
	heightLabel.material.depthTest = false;
	heightLabel.material.opacity = 1;
	heightLabel.visible = false;

	return heightLabel;
}

function createAreaLabel(){
	const areaLabel = new TextSprite('');

	areaLabel.setTextColor({r: 140, g: 250, b: 140, a: 1.0});
	areaLabel.setBorderColor({r: 0, g: 0, b: 0, a: 1.0});
	areaLabel.setBackgroundColor({r: 0, g: 0, b: 0, a: 1.0});
	areaLabel.fontsize = 16;
	areaLabel.material.depthTest = false;
	areaLabel.material.opacity = 1;
	areaLabel.visible = false;
	
	return areaLabel;
}

function createCircleRadiusLabel(){
	const circleRadiusLabel = new TextSprite("");

	circleRadiusLabel.setTextColor({r: 140, g: 250, b: 140, a: 1.0});
	circleRadiusLabel.setBorderColor({r: 0, g: 0, b: 0, a: 1.0});
	circleRadiusLabel.setBackgroundColor({r: 0, g: 0, b: 0, a: 1.0});
	circleRadiusLabel.fontsize = 16;
	circleRadiusLabel.material.depthTest = false;
	circleRadiusLabel.material.opacity = 1;
	circleRadiusLabel.visible = false;
	
	return circleRadiusLabel;
}

function createCircleRadiusLine(){
	const lineGeometry = new LineGeometry();

	lineGeometry.setPositions([
		0, 0, 0,
		0, 0, 0,
	]);

	const lineMaterial = new LineMaterial({ 
		color: 0xff0000, 
		linewidth: 2, 
		resolution:  new THREE.Vector2(1000, 1000),
		gapSize: 1,
		dashed: true,
	});

	lineMaterial.depthTest = false;

	const circleRadiusLine = new Line2(lineGeometry, lineMaterial);
	circleRadiusLine.visible = false;

	return circleRadiusLine;
}

function createCircleLine(){
	const coordinates = [];

	let n = 128;
	for(let i = 0; i <= n; i++){
		let u0 = 2 * Math.PI * (i / n);
		let u1 = 2 * Math.PI * (i + 1) / n;

		let p0 = new THREE.Vector3(
			Math.cos(u0), 
			Math.sin(u0), 
			0
		);

		let p1 = new THREE.Vector3(
			Math.cos(u1), 
			Math.sin(u1), 
			0
		);

		coordinates.push(
			...p0.toArray(),
			...p1.toArray(),
		);
	}

	const geometry = new LineGeometry();
	geometry.setPositions(coordinates);

	const material = new LineMaterial({ 
		color: 0xff0000, 
		dashSize: 5, 
		gapSize: 2,
		linewidth: 2, 
		resolution:  new THREE.Vector2(1000, 1000),
	});

	material.depthTest = false;

	const circleLine = new Line2(geometry, material);
	circleLine.visible = false;
	circleLine.computeLineDistances();

	return circleLine;
}

function createCircleCenter(){
	const sg = new THREE.SphereGeometry(1, 32, 32);
	const sm = new THREE.MeshNormalMaterial();
	
	const circleCenter = new THREE.Mesh(sg, sm);
	circleCenter.visible = false;

	return circleCenter;
}

function createLine(){
	const geometry = new LineGeometry();

	geometry.setPositions([
		0, 0, 0,
		0, 0, 0,
	]);

	const material = new LineMaterial({ 
		color: 0xff0000, 
		linewidth: 2, 
		resolution:  new THREE.Vector2(1000, 1000),
		gapSize: 1,
		dashed: true,
	});

	material.depthTest = false;

	const line = new Line2(geometry, material);

	return line;
}

function createCircle(){

	const coordinates = [];

	let n = 128;
	for(let i = 0; i <= n; i++){
		let u0 = 2 * Math.PI * (i / n);
		let u1 = 2 * Math.PI * (i + 1) / n;

		let p0 = new THREE.Vector3(
			Math.cos(u0), 
			Math.sin(u0), 
			0
		);

		let p1 = new THREE.Vector3(
			Math.cos(u1), 
			Math.sin(u1), 
			0
		);

		coordinates.push(
			...p0.toArray(),
			...p1.toArray(),
		);
	}

	const geometry = new LineGeometry();
	geometry.setPositions(coordinates);

	const material = new LineMaterial({ 
		color: 0xff0000, 
		dashSize: 5, 
		gapSize: 2,
		linewidth: 2, 
		resolution:  new THREE.Vector2(1000, 1000),
	});

	material.depthTest = false;

	const line = new Line2(geometry, material);
	line.computeLineDistances();

	return line;

}

function createAzimuth(){

	const azimuth = {
		label: null,
		center: null,
		target: null,
		north: null,
		centerToNorth: null,
		centerToTarget: null,
		centerToTargetground: null,
		targetgroundToTarget: null,
		circle: null,

		node: null,
	};

	const sg = new THREE.SphereGeometry(1, 32, 32);
	const sm = new THREE.MeshNormalMaterial();

	{
		const label = new TextSprite("");

		label.setTextColor({r: 140, g: 250, b: 140, a: 1.0});
		label.setBorderColor({r: 0, g: 0, b: 0, a: 1.0});
		label.setBackgroundColor({r: 0, g: 0, b: 0, a: 1.0});
		label.fontsize = 16;
		label.material.depthTest = false;
		label.material.opacity = 1;

		azimuth.label = label;
	}

	azimuth.center = new THREE.Mesh(sg, sm);
	azimuth.target = new THREE.Mesh(sg, sm);
	azimuth.north = new THREE.Mesh(sg, sm);
	azimuth.centerToNorth = createLine();
	azimuth.centerToTarget = createLine();
	azimuth.centerToTargetground = createLine();
	azimuth.targetgroundToTarget = createLine();
	azimuth.circle = createCircle();

	azimuth.node = new THREE.Object3D();
	azimuth.node.add(
		azimuth.centerToNorth,
		azimuth.centerToTarget,
		azimuth.centerToTargetground,
		azimuth.targetgroundToTarget,
		azimuth.circle,
		azimuth.label,
		azimuth.center,
		azimuth.target,
		azimuth.north,
	);

	return azimuth;
}

export class Measure extends THREE.Object3D {
	constructor () {
		super();

		this.constructor.counter = (this.constructor.counter === undefined) ? 0 : this.constructor.counter + 1;

		this.name = 'Measure_' + this.constructor.counter;
		this.points = [];
		this._showDistances = true;
		this._showCoordinates = false;
		this._showArea = false;
		this._closed = true;
		this._showAngles = false;
		this._showCircle = false;
		this._showHeight = false;
		this._showEdges = true;
		this._showAzimuth = false;
		this._selected = false;
		this.maxMarkers = Number.MAX_SAFE_INTEGER;

		this.sphereGeometry = new THREE.SphereGeometry(0.4, 10, 10);
		this.color = new THREE.Color(0xff0000);

		this.spheres = [];
		this.edges = [];
		this.sphereLabels = [];
		this.edgeLabels = [];
		this.angleLabels = [];
		this.coordinateLabels = [];

		this.heightEdge = createHeightLine();
		this.heightLabel = createHeightLabel();
		this.areaLabel = createAreaLabel();
		this.circleRadiusLabel = createCircleRadiusLabel();
		this.circleRadiusLine = createCircleRadiusLine();
		this.circleLine = createCircleLine();
		this.circleCenter = createCircleCenter();

		this.azimuth = createAzimuth();

		this.add(this.heightEdge);
		this.add(this.heightLabel);
		this.add(this.areaLabel);
		this.add(this.circleRadiusLabel);
		this.add(this.circleRadiusLine);
		this.add(this.circleLine);
		this.add(this.circleCenter);

		this.add(this.azimuth.node);

		{ // Event Listeners
			// Mirrors Volume.js's selectable-shape pattern, but unlike Volume's no-ops
			// these actually flip _selected - insert/delete-vertex listeners (wired per
			// vertex in _createVertexAt) gate on this.selected, and update() uses it to
			// draw a highlight (since TransformationTool's own selection frame is
			// suppressed for Measure - see TransformationTool.js).
			this.addEventListener('select', e => { this._selected = true; this.update(); });
			this.addEventListener('deselect', e => { this._selected = false; this.update(); });
		}

		{ // Backspace deletes whichever vertex the mouse is currently over - works
		  // regardless of this.selected, since hovering/dragging a point (mouseover,
		  // already set up per-sphere in _createVertexAt) is a more direct "this is the
		  // point I mean" signal than requiring the whole line to be select-clicked
		  // first. _hoveredIndex is maintained by the sphere mouseover/mouseleave
		  // listeners below.
			this._hoveredIndex = null;
			window.addEventListener('keydown', (e) => {
				if (e.key !== 'Backspace' || this._hoveredIndex === null) return;
				if (this.points.length <= 2) return;
				// don't hijack Backspace while the user is typing in an unrelated text
				// field that just happens to sit under the mouse (e.g. EPSG prompt)
				let tag = e.target && e.target.tagName;
				if (tag === 'INPUT' || tag === 'TEXTAREA') return;

				e.preventDefault();
				let index = this._hoveredIndex;
				this._hoveredIndex = null;
				this.removeMarker(index);
			});
		}
	}

	get selected () {
		return this._selected;
	}

	// Required so TransformationTool (which unconditionally reads selected.boundingBox
	// every frame once something is selected) doesn't crash when a Measure is selected.
	// Recomputed on demand rather than cached - point counts are small (tens of
	// vertices) and caching would need invalidating on every update() anyway.
	get boundingBox () {
		if (this.points.length === 0) {
			return new THREE.Box3();
		}

		return new THREE.Box3().setFromPoints(this.points.map(p => p.position));
	}

	createSphereMaterial () {
		let sphereMaterial = new THREE.MeshLambertMaterial({
			//shading: THREE.SmoothShading,
			color: this.color,
			depthTest: false,
			depthWrite: false}
		);

		return sphereMaterial;
	};

	_normalizePoint (point) {
		if (point.x != null) {
			point = {position: point};
		}else if(point instanceof Array){
			point = {position: new THREE.Vector3(...point)};
		}

		return point;
	}

	// Creates one vertex's worth of Object3Ds (sphere/edge/labels) and wires the
	// sphere's drag/drop/hover listeners. Doesn't touch this.points/spheres/edges/etc
	// or call this.add(...) - callers (addMarker/insertMarker) own array placement
	// since append vs. mid-array insert need different splice positions.
	_createVertexAt () {
		let sphere = new THREE.Mesh(this.sphereGeometry, this.createSphereMaterial());

		let edge;
		{
			let lineGeometry = new LineGeometry();
			lineGeometry.setPositions( [
					0, 0, 0,
					0, 0, 0,
			]);

			let lineMaterial = new LineMaterial({
				color: 0xff0000,
				linewidth: 2,
				resolution:  new THREE.Vector2(1000, 1000),
			});

			lineMaterial.depthTest = false;

			edge = new Line2(lineGeometry, lineMaterial);
			edge.visible = true;
		}

		let edgeLabel = new TextSprite();
		edgeLabel.setBorderColor({r: 0, g: 0, b: 0, a: 1.0});
		edgeLabel.setBackgroundColor({r: 0, g: 0, b: 0, a: 1.0});
		edgeLabel.material.depthTest = false;
		edgeLabel.visible = false;
		edgeLabel.fontsize = 16;

		let angleLabel = new TextSprite();
		angleLabel.setBorderColor({r: 0, g: 0, b: 0, a: 1.0});
		angleLabel.setBackgroundColor({r: 0, g: 0, b: 0, a: 1.0});
		angleLabel.fontsize = 16;
		angleLabel.material.depthTest = false;
		angleLabel.material.opacity = 1;
		angleLabel.visible = false;

		let coordinateLabel = new TextSprite();
		coordinateLabel.setBorderColor({r: 0, g: 0, b: 0, a: 1.0});
		coordinateLabel.setBackgroundColor({r: 0, g: 0, b: 0, a: 1.0});
		coordinateLabel.fontsize = 16;
		coordinateLabel.material.depthTest = false;
		coordinateLabel.material.opacity = 1;
		coordinateLabel.visible = false;

		{ // Event Listeners
			let drag = (e) => {
				let I = Utils.getMousePointCloudIntersection(
					e.drag.end,
					e.viewer.scene.getActiveCamera(),
					e.viewer,
					e.viewer.scene.pointclouds,
					{pickClipped: true});

				if (I) {
					let i = this.spheres.indexOf(e.drag.object);
					if (i !== -1) {
						let point = this.points[i];

						// loop through current keys and cleanup ones that will be orphaned
						for (let key of Object.keys(point)) {
							if (!I.point[key]) {
								delete point[key];
							}
						}

						for (let key of Object.keys(I.point).filter(e => e !== 'position')) {
							point[key] = I.point[key];
						}

						let snapped = Utils.snapToNearbyVertex(I.location, e.viewer, {excludeMeasure: this, excludeIndex: i});
						this.setPosition(i, snapped || I.location);
					}
				}
			};

			let drop = e => {
				let i = this.spheres.indexOf(e.drag.object);
				if (i !== -1) {
					this.dispatchEvent({
						'type': 'marker_dropped',
						'measurement': this,
						'index': i
					});
				}
			};

			let mouseover = (e) => {
				e.object.material.emissive.setHex(0x888888);
				this._hoveredIndex = this.spheres.indexOf(e.object);
			};
			let mouseleave = (e) => {
				e.object.material.emissive.setHex(0x000000);
				if (this._hoveredIndex === this.spheres.indexOf(e.object)) {
					this._hoveredIndex = null;
				}
			};

			sphere.addEventListener('drag', drag);
			sphere.addEventListener('drop', drop);
			sphere.addEventListener('mouseover', mouseover);
			sphere.addEventListener('mouseleave', mouseleave);

			// Right-click a vertex of a selected line to delete it. Gated on
			// this.selected so it's a no-op while drawing (not yet selected) or on
			// lines the user hasn't clicked into first - avoids accidental deletes
			// while just navigating/inspecting predictions.
			sphere.addEventListener('mouseup', (e) => {
				if (!this.selected || e.button !== THREE.MOUSE.RIGHT) return;
				if (this.points.length <= 2) return;

				e.consume();
				let i = this.spheres.indexOf(e.target);
				if (i !== -1) this.removeMarker(i);
			});

			// Click a segment of a selected line to insert a vertex there. Reads the
			// point/edge index from the InputHandler's own hoveredElements (populated
			// by the preceding mousemove's raycast) rather than the mouseup payload,
			// which only carries {type, viewer, consume} - see InputHandler.js onMouseUp.
			edge.addEventListener('mouseup', (e) => {
				if (!this.selected || e.button !== THREE.MOUSE.LEFT) return;

				let hovered = e.viewer.inputHandler.hoveredElements.find(h => h.object === e.target);
				if (!hovered) return;

				e.consume();
				let i = this.edges.indexOf(e.target);
				if (i === -1) return;

				let position = Utils.snapToNearbyVertex(hovered.point, e.viewer, {excludeMeasure: this}) || hovered.point.clone();
				this.insertMarker(i + 1, position);
			});
		}

		return {sphere, edge, edgeLabel, angleLabel, coordinateLabel};
	}

	addMarker (point) {
		point = this._normalizePoint(point);
		this.points.push(point);

		let {sphere, edge, edgeLabel, angleLabel, coordinateLabel} = this._createVertexAt();

		this.add(sphere, edge, edgeLabel, angleLabel, coordinateLabel);
		this.spheres.push(sphere);
		this.edges.push(edge);
		this.edgeLabels.push(edgeLabel);
		this.angleLabels.push(angleLabel);
		this.coordinateLabels.push(coordinateLabel);

		let event = {
			type: 'marker_added',
			measurement: this,
			sphere: sphere
		};
		this.dispatchEvent(event);

		this.setMarker(this.points.length - 1, point);
	};

	// Inserts a new vertex at `index`, splitting the edge that used to run from
	// points[index - 1] to points[index]. update() fully re-derives all edge
	// geometry/labels from this.points each call, so no manual geometry patching
	// is needed here beyond getting the array insert position right.
	insertMarker (index, point) {
		point = this._normalizePoint(point);
		this.points.splice(index, 0, point);

		let {sphere, edge, edgeLabel, angleLabel, coordinateLabel} = this._createVertexAt();

		this.add(sphere, edge, edgeLabel, angleLabel, coordinateLabel);
		this.spheres.splice(index, 0, sphere);
		this.edges.splice(index, 0, edge);
		this.edgeLabels.splice(index, 0, edgeLabel);
		this.angleLabels.splice(index, 0, angleLabel);
		this.coordinateLabels.splice(index, 0, coordinateLabel);

		this.dispatchEvent({
			type: 'marker_added',
			measurement: this,
			sphere: sphere
		});
		this.dispatchEvent({type: 'marker_inserted', measurement: this, index: index});

		this.update();
	};

	removeMarker (index) {
		this.points.splice(index, 1);

		this.remove(this.spheres[index]);

		let edgeIndex = (index === 0) ? 0 : (index - 1);
		this.remove(this.edges[edgeIndex]);
		this.edges.splice(edgeIndex, 1);

		this.remove(this.edgeLabels[edgeIndex]);
		this.edgeLabels.splice(edgeIndex, 1);
		this.coordinateLabels.splice(index, 1);

		this.remove(this.angleLabels[index]);
		this.angleLabels.splice(index, 1);

		this.spheres.splice(index, 1);

		this.update();

		this.dispatchEvent({type: 'marker_removed', measurement: this});
	};

	setMarker (index, point) {
		this.points[index] = point;

		let event = {
			type: 'marker_moved',
			measure:	this,
			index:	index,
			position: point.position.clone()
		};
		this.dispatchEvent(event);

		this.update();
	}

	setPosition (index, position) {
		let point = this.points[index];
		point.position.copy(position);

		let event = {
			type: 'marker_moved',
			measure:	this,
			index:	index,
			position: position.clone()
		};
		this.dispatchEvent(event);

		this.update();
	};

	getArea () {
		let area = 0;
		let j = this.points.length - 1;

		for (let i = 0; i < this.points.length; i++) {
			let p1 = this.points[i].position;
			let p2 = this.points[j].position;
			area += (p2.x + p1.x) * (p1.y - p2.y);
			j = i;
		}

		return Math.abs(area / 2);
	};

	getTotalDistance () {
		if (this.points.length === 0) {
			return 0;
		}

		let distance = 0;

		for (let i = 1; i < this.points.length; i++) {
			let prev = this.points[i - 1].position;
			let curr = this.points[i].position;
			let d = prev.distanceTo(curr);

			distance += d;
		}

		if (this.closed && this.points.length > 1) {
			let first = this.points[0].position;
			let last = this.points[this.points.length - 1].position;
			let d = last.distanceTo(first);

			distance += d;
		}

		return distance;
	}

	getAngleBetweenLines (cornerPoint, point1, point2) {
		let v1 = new THREE.Vector3().subVectors(point1.position, cornerPoint.position);
		let v2 = new THREE.Vector3().subVectors(point2.position, cornerPoint.position);

		// avoid the error printed by threejs if denominator is 0
		const denominator = Math.sqrt( v1.lengthSq() * v2.lengthSq() );
		if(denominator === 0){
			return 0;
		}else{
			return v1.angleTo(v2);
		}
	};

	getAngle (index) {
		if (this.points.length < 3 || index >= this.points.length) {
			return 0;
		}

		let previous = (index === 0) ? this.points[this.points.length - 1] : this.points[index - 1];
		let point = this.points[index];
		let next = this.points[(index + 1) % (this.points.length)];

		return this.getAngleBetweenLines(point, previous, next);
	}

	// updateAzimuth(){
	// 	// if(this.points.length !== 2){
	// 	// 	return;
	// 	// }

	// 	// const azimuth = this.azimuth;

	// 	// const [p0, p1] = this.points;

	// 	// const r = p0.position.distanceTo(p1.position);
		
	// }

	update () {
		if (this.points.length === 0) {
			return;
		} else if (this.points.length === 1) {
			let point = this.points[0];
			let position = point.position;
			this.spheres[0].position.copy(position);

			{ // coordinate labels
				let coordinateLabel = this.coordinateLabels[0];
				
				let msg = position.toArray().map(p => Utils.addCommas(p.toFixed(2))).join(" / ");
				coordinateLabel.setText(msg);

				coordinateLabel.visible = this.showCoordinates;
			}

			return;
		}

		let lastIndex = this.points.length - 1;

		let centroid = new THREE.Vector3();
		for (let i = 0; i <= lastIndex; i++) {
			let point = this.points[i];
			centroid.add(point.position);
		}
		centroid.divideScalar(this.points.length);

		for (let i = 0; i <= lastIndex; i++) {
			let index = i;
			let nextIndex = (i + 1 > lastIndex) ? 0 : i + 1;
			let previousIndex = (i === 0) ? lastIndex : i - 1;

			let point = this.points[index];
			let nextPoint = this.points[nextIndex];
			let previousPoint = this.points[previousIndex];

			let sphere = this.spheres[index];

			// spheres
			sphere.position.copy(point.position);
			sphere.material.color = this.color;
			sphere.material.emissive.setHex(this.selected ? 0x333333 : 0x000000);

			{ // edges
				let edge = this.edges[index];

				if(this.selected){
					edge.material.color = this.color.clone().lerp(new THREE.Color(0xffff00), 0.5);
					edge.material.linewidth = 4;
				}else{
					edge.material.color = this.color;
					edge.material.linewidth = 2;
				}

				edge.position.copy(point.position);

				edge.geometry.setPositions([
					0, 0, 0,
					...nextPoint.position.clone().sub(point.position).toArray(),
				]);

				edge.geometry.verticesNeedUpdate = true;
				edge.geometry.computeBoundingSphere();
				edge.computeLineDistances();
				edge.visible = index < lastIndex || this.closed;
				
				if(!this.showEdges){
					edge.visible = false;
				}
			}

			{ // edge labels
				let edgeLabel = this.edgeLabels[i];

				let center = new THREE.Vector3().add(point.position);
				center.add(nextPoint.position);
				center = center.multiplyScalar(0.5);
				let distance = point.position.distanceTo(nextPoint.position);

				edgeLabel.position.copy(center);

				let suffix = "";
				if(this.lengthUnit != null && this.lengthUnitDisplay != null){
					distance = distance / this.lengthUnit.unitspermeter * this.lengthUnitDisplay.unitspermeter;  //convert to meters then to the display unit
					suffix = this.lengthUnitDisplay.code;
				}

				let txtLength = Utils.addCommas(distance.toFixed(2));
				edgeLabel.setText(`${txtLength} ${suffix}`);
				edgeLabel.visible = this.showDistances && (index < lastIndex || this.closed) && this.points.length >= 2 && distance > 0;
			}

			{ // angle labels
				let angleLabel = this.angleLabels[i];
				let angle = this.getAngleBetweenLines(point, previousPoint, nextPoint);

				let dir = nextPoint.position.clone().sub(previousPoint.position);
				dir.multiplyScalar(0.5);
				dir = previousPoint.position.clone().add(dir).sub(point.position).normalize();

				let dist = Math.min(point.position.distanceTo(previousPoint.position), point.position.distanceTo(nextPoint.position));
				dist = dist / 9;

				let labelPos = point.position.clone().add(dir.multiplyScalar(dist));
				angleLabel.position.copy(labelPos);

				let msg = Utils.addCommas((angle * (180.0 / Math.PI)).toFixed(1)) + '\u00B0';
				angleLabel.setText(msg);

				angleLabel.visible = this.showAngles && (index < lastIndex || this.closed) && this.points.length >= 3 && angle > 0;
			}
		}

		{ // update height stuff
			let heightEdge = this.heightEdge;
			heightEdge.visible = this.showHeight;
			this.heightLabel.visible = this.showHeight;

			if (this.showHeight) {
				let sorted = this.points.slice().sort((a, b) => a.position.z - b.position.z);
				let lowPoint = sorted[0].position.clone();
				let highPoint = sorted[sorted.length - 1].position.clone();
				let min = lowPoint.z;
				let max = highPoint.z;
				let height = max - min;

				let start = new THREE.Vector3(highPoint.x, highPoint.y, min);
				let end = new THREE.Vector3(highPoint.x, highPoint.y, max);

				heightEdge.position.copy(lowPoint);

				heightEdge.geometry.setPositions([
					0, 0, 0,
					...start.clone().sub(lowPoint).toArray(),
					...start.clone().sub(lowPoint).toArray(),
					...end.clone().sub(lowPoint).toArray(),
				]);

				heightEdge.geometry.verticesNeedUpdate = true;
				// heightEdge.geometry.computeLineDistances();
				// heightEdge.geometry.lineDistancesNeedUpdate = true;
				heightEdge.geometry.computeBoundingSphere();
				heightEdge.computeLineDistances();

				// heightEdge.material.dashSize = height / 40;
				// heightEdge.material.gapSize = height / 40;

				let heightLabelPosition = start.clone().add(end).multiplyScalar(0.5);
				this.heightLabel.position.copy(heightLabelPosition);

				let suffix = "";
				if(this.lengthUnit != null && this.lengthUnitDisplay != null){
					height = height / this.lengthUnit.unitspermeter * this.lengthUnitDisplay.unitspermeter;  //convert to meters then to the display unit
					suffix = this.lengthUnitDisplay.code;
				}

				let txtHeight = Utils.addCommas(height.toFixed(2));
				let msg = `${txtHeight} ${suffix}`;
				this.heightLabel.setText(msg);
			}
		}

		{ // update circle stuff
			const circleRadiusLabel = this.circleRadiusLabel;
			const circleRadiusLine = this.circleRadiusLine;
			const circleLine = this.circleLine;
			const circleCenter = this.circleCenter;

			const circleOkay = this.points.length === 3;

			circleRadiusLabel.visible = this.showCircle && circleOkay;
			circleRadiusLine.visible = this.showCircle && circleOkay;
			circleLine.visible = this.showCircle && circleOkay;
			circleCenter.visible = this.showCircle && circleOkay;

			if(this.showCircle && circleOkay){

				const A = this.points[0].position;
				const B = this.points[1].position;
				const C = this.points[2].position;
				const AB = B.clone().sub(A);
				const AC = C.clone().sub(A);
				const N = AC.clone().cross(AB).normalize();

				const center = Potree.Utils.computeCircleCenter(A, B, C);
				const radius = center.distanceTo(A);


				const scale = radius / 20;
				circleCenter.position.copy(center);
				circleCenter.scale.set(scale, scale, scale);

				//circleRadiusLine.geometry.vertices[0].set(0, 0, 0);
				//circleRadiusLine.geometry.vertices[1].copy(B.clone().sub(center));

				circleRadiusLine.geometry.setPositions( [
					0, 0, 0,
					...B.clone().sub(center).toArray()
				] );

				circleRadiusLine.geometry.verticesNeedUpdate = true;
				circleRadiusLine.geometry.computeBoundingSphere();
				circleRadiusLine.position.copy(center);
				circleRadiusLine.computeLineDistances();

				const target = center.clone().add(N);
				circleLine.position.copy(center);
				circleLine.scale.set(radius, radius, radius);
				circleLine.lookAt(target);
				
				circleRadiusLabel.visible = true;
				circleRadiusLabel.position.copy(center.clone().add(B).multiplyScalar(0.5));
				circleRadiusLabel.setText(`${radius.toFixed(3)}`);

			}
		}

		{ // update area label
			this.areaLabel.position.copy(centroid);
			this.areaLabel.visible = this.showArea && this.points.length >= 3;
			let area = this.getArea();

			let suffix = "";
			if(this.lengthUnit != null && this.lengthUnitDisplay != null){
				area = area / Math.pow(this.lengthUnit.unitspermeter, 2) * Math.pow(this.lengthUnitDisplay.unitspermeter, 2);  //convert to square meters then to the square display unit
				suffix = this.lengthUnitDisplay.code;
			}

			let txtArea = Utils.addCommas(area.toFixed(1));
			let msg =  `${txtArea} ${suffix}\u00B2`;
			this.areaLabel.setText(msg);
		}

		// this.updateAzimuth();
	};

	// Only used to let a click on an edge select the whole Measure - mirrors
	// BoxVolume.raycast (Volume.js), reporting a hit with object:this instead of the
	// child mesh, in a local array so unrelated objects' entries in the shared
	// `intersects` aren't touched. Deliberately does NOT also test spheres: spheres
	// already raycast themselves natively (object:the sphere, for drag/hover/delete)
	// and since Measure.raycast()'s hit would land at the exact same point/distance,
	// the two would tie and - depending on scene-traversal order - could shadow the
	// sphere's own mouseover/mouseleave dispatch (InputHandler only fires
	// mouseover/mouseleave for the single closest hit overall), breaking hover
	// highlighting and hovered-vertex tracking. Leaving spheres out means clicking
	// exactly on a vertex doesn't select the line (clicking the edge next to it
	// does), which is an acceptable gap given vertices are primarily for
	// dragging/deleting, not selecting.
	raycast (raycaster, intersects) {
		let local = [];

		for (let i = 0; i < this.edges.length; i++) {
			let edge = this.edges[i];
			if (!edge.visible) continue;

			let edgeHits = [];
			edge.raycast(raycaster, edgeHits);
			for (let hit of edgeHits) {
				local.push({distance: hit.distance, object: this, point: hit.point.clone()});
			}
		}

		if (local.length === 0) return;

		local.sort((a, b) => a.distance - b.distance);
		intersects.push(local[0]);
	};

	get showCoordinates () {
		return this._showCoordinates;
	}

	set showCoordinates (value) {
		this._showCoordinates = value;
		this.update();
	}

	get showAngles () {
		return this._showAngles;
	}

	set showAngles (value) {
		this._showAngles = value;
		this.update();
	}

	get showCircle () {
		return this._showCircle;
	}

	set showCircle (value) {
		this._showCircle = value;
		this.update();
	}

	get showAzimuth(){
		return this._showAzimuth;
	}

	set showAzimuth(value){
		this._showAzimuth = value;
		this.update();
	}

	get showEdges () {
		return this._showEdges;
	}

	set showEdges (value) {
		this._showEdges = value;
		this.update();
	}

	get showHeight () {
		return this._showHeight;
	}

	set showHeight (value) {
		this._showHeight = value;
		this.update();
	}

	get showArea () {
		return this._showArea;
	}

	set showArea (value) {
		this._showArea = value;
		this.update();
	}

	get closed () {
		return this._closed;
	}

	set closed (value) {
		this._closed = value;
		this.update();
	}

	get showDistances () {
		return this._showDistances;
	}

	set showDistances (value) {
		this._showDistances = value;
		this.update();
	}

}
