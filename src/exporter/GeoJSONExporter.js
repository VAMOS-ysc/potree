/**
 *
 * @author sigeom sa / http://sigeom.ch
 * @author Ioda-Net Sàrl / https://www.ioda-net.ch/
 * @author Markus Schütz / http://potree.org
 *
 */

import {Measure} from "../utils/Measure.js";

export class GeoJSONExporter{

	static measurementToFeatures (measurement) {
		let coords = measurement.points.map(e => e.position.toArray());

		let features = [];

		// Kind/Type mirror the 정밀도로지도 B2_SURFACELINEMARK/B3_SURFACEMARK field
		// names ml/rasterize.py's DEFAULT_KIND_TO_CLASS and CROSSWALK_KIND_VALUE
		// key off of - as strings, since rasterize.py compares them with a SQL
		// "field = 'value'" -where clause. Without these, an exported Lane/Area
		// selection can't be fed back into rasterize.py --lines/--crosswalks to
		// regenerate training masks after correcting a model prediction.
		let laneProperties = measurement.isLane
			? {laneType: measurement.laneType, laneColor: measurement.laneColor,
			   Kind: measurement.laneType === 'stop_line' ? '530' : '503'}
			: {};
		let crosswalkProperties = measurement.isCrosswalk
			? {Type: '5', Kind: '5321'}
			: {};

		if (coords.length === 1) {
			let feature = {
				type: 'Feature',
				geometry: {
					type: 'Point',
					coordinates: coords[0]
				},
				properties: {
					name: measurement.name,
					...laneProperties
				}
			};
			features.push(feature);
		} else if (coords.length > 1 && !measurement.closed) {
			let object = {
				'type': 'Feature',
				'geometry': {
					'type': 'LineString',
					'coordinates': coords
				},
				'properties': {
					name: measurement.name,
					...laneProperties
				}
			};

			features.push(object);
		} else if (coords.length > 1 && measurement.closed) {
			let object = {
				'type': 'Feature',
				'geometry': {
					'type': 'Polygon',
					'coordinates': [[...coords, coords[0]]]
				},
				'properties': {
					name: measurement.name,
					...crosswalkProperties
				}
			};
			features.push(object);
		}

		if (measurement.showDistances) {
			measurement.edgeLabels.forEach((label) => {
				let labelPoint = {
					type: 'Feature',
					geometry: {
						type: 'Point',
						coordinates: label.position.toArray()
					},
					properties: {
						distance: label.text
					}
				};
				features.push(labelPoint);
			});
		}

		if (measurement.showArea) {
			let point = measurement.areaLabel.position;
			let labelArea = {
				type: 'Feature',
				geometry: {
					type: 'Point',
					coordinates: point.toArray()
				},
				properties: {
					area: measurement.areaLabel.text
				}
			};
			features.push(labelArea);
		}

		return features;
	}

	static toString (measurements) {
		if (!(measurements instanceof Array)) {
			measurements = [measurements];
		}

		measurements = measurements.filter(m => m instanceof Measure);

		let features = [];
		for (let measure of measurements) {
			let f = GeoJSONExporter.measurementToFeatures(measure);

			features = features.concat(f);
		}

		let geojson = {
			'type': 'FeatureCollection',
			'features': features
		};

		return JSON.stringify(geojson, null, '\t');
	}

}
