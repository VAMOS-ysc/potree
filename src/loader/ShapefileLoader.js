
import * as THREE from "../../libs/three.js/build/three.module.js";
import {Line2} from "../../libs/three.js/lines/Line2.js";
import {LineGeometry} from "../../libs/three.js/lines/LineGeometry.js";
import {LineMaterial} from "../../libs/three.js/lines/LineMaterial.js";

// Shared circular-dot texture for Point features, generated once and reused across
// all ShapefileLoader instances/loads. A world-space sphere (the original approach)
// shrinks to sub-pixel and disappears once zoomed out to see a whole road corridor,
// since it's sized in meters; lines don't have this problem because LineMaterial's
// linewidth is already defined in screen pixels. This texture backs a THREE.Points
// marker with sizeAttenuation disabled, which is the Points equivalent - constant
// on-screen size at any zoom/distance, matching how the lines already behave.
let pointDotTexture = null;
function getPointDotTexture(){
	if(pointDotTexture !== null){
		return pointDotTexture;
	}

	// a dark outline around the fill keeps the dot legible against both a bright
	// point cloud and a same-colored line/polygon overlay sitting right behind it
	const size = 64;
	const canvas = document.createElement("canvas");
	canvas.width = size;
	canvas.height = size;
	const ctx = canvas.getContext("2d");
	ctx.beginPath();
	ctx.arc(size / 2, size / 2, size / 2 - 4, 0, 2 * Math.PI);
	ctx.fillStyle = "white";
	ctx.fill();
	ctx.lineWidth = 5;
	ctx.strokeStyle = "black";
	ctx.stroke();

	pointDotTexture = new THREE.CanvasTexture(canvas);
	return pointDotTexture;
}

export class ShapefileLoader{

	constructor(){
		this.transform = null;
		// world-space Z to place all loaded geometry at - shapefiles have no elevation
		// of their own here, so this must be set to roughly the point cloud's ground
		// height for the overlay to actually align with it (default 20 only matched
		// the original low-elevation demo dataset)
		this.z = 20;
		// on-screen size of the dot used for Point features, in pixels (not world
		// units - see getPointDotTexture() above for why)
		this.pointPixelSize = 14;
		// Point features get their own color, independent of the line/polygon color
		// (matLine.color, red by default) - every overlay layer otherwise renders in
		// the same red, so a small red dot is indistinguishable from the red lines/
		// polygons sitting right behind it. Yellow reads clearly against both.
		this.pointColor = 0xffdd00;
	}

	async load(path){

		const matLine = new LineMaterial( {
			color: 0xff0000,
			linewidth: 3, // in pixels
			resolution:  new THREE.Vector2(1000, 1000),
			dashed: false
		} );

		const features = await this.loadShapefileFeatures(path);
		const node = new THREE.Object3D();
		
		for(const feature of features){
			const fnode = this.featureToSceneNode(feature, matLine);
			node.add(fnode);
		}

		let setResolution = (x, y) => {
			matLine.resolution.set(x, y);
		};

		const result = {
			features: features,
			node: node,
			setResolution: setResolution,
		};

		return result;
	}

	featureToSceneNode(feature, matLine){
		let geometry = feature.geometry;

		let color = new THREE.Color(1, 1, 1);

		// shapefiles with elevation/measure data report types like "PointZ"/"LineStringZ"/
		// "PolygonZ" (or "M"/"ZM") - these carry the same [x,y] (+ extra) coordinate shape,
		// so normalize to the base type rather than silently dropping every feature
		const baseType = geometry.type.replace(/(Z|M|ZM)$/, "");

		let transform = this.transform;
		if(transform === null){
			transform = {forward: (v) => v};
		}
		
		if(baseType === "Point"){
			let pg = new THREE.BufferGeometry();
			pg.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0], 3));

			// sizeAttenuation: false keeps this a constant pixel size regardless of
			// camera distance/zoom (see getPointDotTexture() above); depthTest off +
			// high renderOrder keeps it from being buried inside/behind the point
			// cloud's own points at the same location
			let pm = new THREE.PointsMaterial({
				color: this.pointColor,
				size: this.pointPixelSize,
				sizeAttenuation: false,
				map: getPointDotTexture(),
				alphaTest: 0.5,
				depthTest: false,
			});
			let s = new THREE.Points(pg, pm);
			s.renderOrder = 10;

			let [long, lat, elevation] = geometry.coordinates;
			let z = elevation !== undefined ? elevation : this.z;
			let pos = transform.forward([long, lat]);

			s.position.set(...pos, z);

			return s;
		}else if(baseType === "LineString"){
			let coordinates = [];

			let min = new THREE.Vector3(Infinity, Infinity, Infinity);
			for(let i = 0; i < geometry.coordinates.length; i++){
				let [long, lat, elevation] = geometry.coordinates[i];
				// use the feature's own elevation when present (e.g. draped onto real
				// ground height server-side) instead of a single flat this.z for everything
				let z = elevation !== undefined ? elevation : this.z;
				let pos = transform.forward([long, lat]);

				min.x = Math.min(min.x, pos[0]);
				min.y = Math.min(min.y, pos[1]);
				min.z = Math.min(min.z, z);

				coordinates.push(...pos, z);
				if(i > 0 && i < geometry.coordinates.length - 1){
					coordinates.push(...pos, z);
				}
			}
			
			for(let i = 0; i < coordinates.length; i += 3){
				coordinates[i+0] -= min.x;
				coordinates[i+1] -= min.y;
				coordinates[i+2] -= min.z;
			}
			
			const lineGeometry = new LineGeometry();
			lineGeometry.setPositions( coordinates );

			const line = new Line2( lineGeometry, matLine );
			line.computeLineDistances();
			line.scale.set( 1, 1, 1 );
			line.position.copy(min);
			
			return line;
		}else if(baseType === "Polygon"){
			for(let pc of geometry.coordinates){
				let coordinates = [];
				
				let min = new THREE.Vector3(Infinity, Infinity, Infinity);
				for(let i = 0; i < pc.length; i++){
					let [long, lat, elevation] = pc[i];
					let z = elevation !== undefined ? elevation : this.z;
					let pos = transform.forward([long, lat]);

					min.x = Math.min(min.x, pos[0]);
					min.y = Math.min(min.y, pos[1]);
					min.z = Math.min(min.z, z);

					coordinates.push(...pos, z);
					if(i > 0 && i < pc.length - 1){
						coordinates.push(...pos, z);
					}
				}
				
				for(let i = 0; i < coordinates.length; i += 3){
					coordinates[i+0] -= min.x;
					coordinates[i+1] -= min.y;
					coordinates[i+2] -= min.z;
				}

				const lineGeometry = new LineGeometry();
				lineGeometry.setPositions( coordinates );

				const line = new Line2( lineGeometry, matLine );
				line.computeLineDistances();
				line.scale.set( 1, 1, 1 );
				line.position.copy(min);
				
				return line;
			}
		}else{
			console.log("unhandled feature: ", feature);
		}
	}

	async loadShapefileFeatures(file){
		let features = [];

		let source = await shapefile.open(file);

		while(true){
			let result = await source.read();

			if (result.done) {
				break;
			}

			if (result.value && result.value.type === 'Feature' && result.value.geometry !== undefined) {
				features.push(result.value);
			}
		}

		return features;
	}

};

