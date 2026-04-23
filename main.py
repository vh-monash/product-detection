import os
import tempfile
from io import BytesIO

import fitz
import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

from detection.extract import extract_and_crop_by_mask

# --- Config ---

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

api_key = os.environ["API_KEY"]
url = os.environ["URL"]
args = {"conf": 0.8, "iou": 0.7, "imgsz": 640}  # YOLO inference parameters

# --- App setup ---

app = Flask(__name__)
# Allow both localhost and 127.0.0.1 — browsers treat them as different origins
CORS(app, origins=["http://localhost:5500", "http://127.0.0.1:5500"])
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# --- Helpers ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Routes ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    ext = file.filename.lower().rsplit('.', 1)[1]

    if ext == 'pdf':
        return process_pdf(file)
    else:
        return process_image(file)


# --- File processors ---

def process_pdf(file):
    """Rasterise the first page of a PDF and run it through the detection pipeline."""
    try:
        pdf_bytes = BytesIO(file.read())
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        if pdf_document.page_count == 0:
            return jsonify({'error': 'PDF has no pages'}), 400

        page = pdf_document[0]
        # PDF units are 72-point; scale to 300 DPI for a clean raster
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return send_to_api_and_return(image, "processed_page.png")

    except Exception as e:
        return jsonify({'error': f'PDF processing failed: {str(e)}'}), 500


def process_image(file):
    """Open an image file and run it through the detection pipeline."""
    try:
        image = Image.open(file.stream).convert("RGB")
        original_name = file.filename.rsplit('.', 1)[0]
        return send_to_api_and_return(image, f"{original_name}_processed.png")

    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500


def send_to_api_and_return(image, output_filename):
    """
    Send image to the detection API, apply the returned masks locally,
    and stream the result back to the client.
    """
    try:
        # Write to a temp file — it serves as the shared source for both
        # the multipart API upload and the subsequent local extraction step.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            image.save(tmpfile.name, "PNG")
            tmpfile_path = tmpfile.name

        try:
            with open(tmpfile_path, "rb") as f:
                response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=args,
                    files={"file": f},
                )

            if response.status_code != 200:
                return jsonify({'error': f'API returned status {response.status_code}'}), response.status_code

            results = response.json()
            images = results.get("images", [])

            original_image = Image.open(tmpfile_path).convert("RGB")
            processed_image = extract_and_crop_by_mask(original_image, images)

            output_buffer = BytesIO()
            processed_image.save(output_buffer, format='PNG')
            output_buffer.seek(0)

            return send_file(
                output_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name=output_filename
            )

        finally:
            if os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)

    except Exception as e:
        return jsonify({'error': 'Processing failed', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
