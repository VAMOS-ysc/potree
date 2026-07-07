

import * as THREE from "../../../libs/three.js/build/three.module.js";
import {DistancePanel} from "./DistancePanel.js";

const LANE_COLORS = {
	white: 0xffffff,
	yellow: 0xffdd00,
};

export class LanePanel extends DistancePanel{
	constructor(viewer, measurement, propertiesPanel){
		super(viewer, measurement, propertiesPanel);

		const elLaneAttributes = $(`
			<div class="measurement_content selectable">
				<label>
					Lane Type:
					<select name="lane_type">
						<option value="solid">solid</option>
						<option value="dashed">dashed</option>
						<option value="stop_line">stop line</option>
					</select>
				</label>
				<br>
				<label>
					Lane Color:
					<select name="lane_color">
						<option value="white">white</option>
						<option value="yellow">yellow</option>
					</select>
				</label>
			</div>
		`);

		this.elContent.prepend(elLaneAttributes);

		const elLaneType = elLaneAttributes.find("select[name=lane_type]");
		const elLaneColor = elLaneAttributes.find("select[name=lane_color]");

		elLaneType.val(measurement.laneType);
		elLaneColor.val(measurement.laneColor);

		elLaneType.change(() => {
			measurement.laneType = elLaneType.val();
		});

		elLaneColor.change(() => {
			measurement.laneColor = elLaneColor.val();
			measurement.color = new THREE.Color(LANE_COLORS[measurement.laneColor]);
			measurement.update();
		});
	}
};
