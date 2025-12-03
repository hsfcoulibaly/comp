# app.py
from flask import Flask, request, jsonify, render_template
import math
import numpy as np
import os

app = Flask(__name__)

# --- Load Camera Calibration Data at Startup ---
calibration_file = 'camera_calibration.npz'
if not os.path.exists(calibration_file):
    raise FileNotFoundError(
        "Calibration file 'camera_calibration.npz' not found. "
        "Please run the calibrate_camera.py script first."
    )

calibration_data = np.load(calibration_file)
camera_matrix = calibration_data['camera_matrix']

# Extract the calibrated focal lengths from the camera matrix
CALIBRATED_FX = camera_matrix[0, 0]  # Value from mtx[0][0]
CALIBRATED_FY = camera_matrix[1, 1]  # Value from mtx[1][1]

print("--- Camera Intrinsics Loaded ---")
print(f"Focal Length fx: {CALIBRATED_FX:.2f}")
print(f"Focal Length fy: {CALIBRATED_FY:.2f}")
print("--------------------------------")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate_distance():
    data = request.json
    points = data.get('points')
    displayed_dims = data.get('displayedDimensions')
    original_dims = data.get('originalDimensions')
    camera_distance = float(data.get('cameraDistance', 61))

    if not points or len(points) < 2 or not displayed_dims or not original_dims:
        return jsonify({'error': 'Insufficient data. Requires 2 points, displayed dimensions, and original dimensions.'}), 400

    scale_x = displayed_dims['width'] / original_dims['width']
    scale_y = displayed_dims['height'] / original_dims['height']

    p1_displayed = points[0]
    p2_displayed = points[1]

    pixel_diff_x_displayed = p2_displayed['x'] - p1_displayed['x']
    pixel_diff_y_displayed = p2_displayed['y'] - p1_displayed['y']

    pixel_diff_x_original = pixel_diff_x_displayed / scale_x
    pixel_diff_y_original = pixel_diff_y_displayed / scale_y

    real_world_dist_x = (pixel_diff_x_original * camera_distance) / CALIBRATED_FX
    real_world_dist_y = (pixel_diff_y_original * camera_distance) / CALIBRATED_FY

    total_real_world_distance = math.sqrt(real_world_dist_x**2 + real_world_dist_y**2)

    result = {
        'realWorldDistanceX': f"{real_world_dist_x:.4f}",
        'realWorldDistanceY': f"{real_world_dist_y:.4f}",
        'totalRealWorldDistance': f"{total_real_world_distance:.4f}",
        'details': {
            'point1_displayed': p1_displayed,
            'point2_displayed': p2_displayed,
            'pixel_diff_original_x': f"{pixel_diff_x_original:.2f}",
            'pixel_diff_original_y': f"{pixel_diff_y_original:.2f}",
            'calibrated_fx': CALIBRATED_FX,
            'calibrated_fy': CALIBRATED_FY,
            'camera_distance': camera_distance,
            'scale_factors': {'x': f"{scale_x:.4f}", 'y': f"{scale_y:.4f}"}
        }
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)