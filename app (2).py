# ... existing imports (Flask, request, jsonify) ...
from werkzeug.utils import secure_filename
import os
import time
from
from utils.template_matcher import find_best_match_and_blur

# from utils.fourier_solver import deblur_with_fourier_inverse # Task 2 is typically a script, but can be integrated

# Configuration for image uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Define the path to the template directory
TEMPLATE_DIR = os.path.join(app.root_path, 'static', 'templates')


@app.route('/template_matching', methods=['POST'])
def template_matching():
    """Handles object detection, blurring the detected region, and returns the modified image."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        # Save the uploaded file securely
        filename = secure_filename(f"{time.time()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image using the template matcher
        processed_img, message = find_best_match_and_blur(filepath, TEMPLATE_DIR)

        # Clean up the original uploaded image file
        os.remove(filepath)

        if processed_img is None:
            return jsonify({'status': 'error', 'message': message}), 500

        # Save the processed image temporarily to send back to the frontend
        output_filename = f"processed_{filename}"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_filepath, processed_img)

        # Return the URL of the processed image
        return jsonify({
            'status': 'success',
            'message': message,
            'image_url': f'/uploads/{output_filename}'
        })


# Route to serve temporary uploaded and processed files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ... The rest of the Assignment 1 code remains ...