import logging
import os
import tempfile
from io import BytesIO

import fitz
import requests
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image

from detection.extract import extract_and_crop_by_mask

# --- Config ---

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
MAX_FILE_SIZE      = 50 * 1024 * 1024  # 50 MB
API_TIMEOUT        = 60                 # seconds before giving up on the detection API

api_key = os.environ["API_KEY"]  # fail fast at startup if missing
api_url = os.environ["URL"]
args    = {"conf": 0.8, "iou": 0.7, "imgsz": 640}  # YOLO inference parameters

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
)
log = logging.getLogger(__name__)

# --- App setup ---

app = Flask(__name__)

# CORS origins are configurable so the same backend can serve different frontends
# across environments without a code change.
_raw_origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:5500,http://127.0.0.1:5500')
CORS(app, origins=_raw_origins.split(','))

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# --- Security headers (applied to every response) ---

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options']        = 'DENY'
    return response


# --- Error handlers ---

@app.errorhandler(413)
def file_too_large(_e):
    mb = MAX_FILE_SIZE // (1024 * 1024)
    return jsonify({'error': f'File exceeds the {mb} MB size limit'}), 413

@app.errorhandler(404)
def not_found(_e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    log.error('Unhandled exception: %s', e, exc_info=True)
    return jsonify({'error': 'An unexpected server error occurred'}), 500


# --- Routes ---

@app.route('/')
def frontend():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory(BASE_DIR, 'styles.css')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not _allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    log.info('Received upload: %s', file.filename)

    ext = file.filename.lower().rsplit('.', 1)[1]
    return process_pdf(file) if ext == 'pdf' else process_image(file)


# --- File processors ---

def process_pdf(file):
    """Rasterise the first page of a PDF and run it through the detection pipeline."""
    try:
        pdf_bytes   = BytesIO(file.read())
        pdf_doc     = fitz.open(stream=pdf_bytes, filetype="pdf")

        if pdf_doc.page_count == 0:
            return jsonify({'error': 'PDF has no pages'}), 400

        page = pdf_doc[0]
        # PDF units are 72-point; scale to 300 DPI for a clean raster
        pix   = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return _send_to_api_and_return(image, "processed_page.png")

    except Exception as e:
        log.error('PDF processing failed: %s', e, exc_info=True)
        return jsonify({'error': 'PDF processing failed'}), 500


def process_image(file):
    """Open an image file and run it through the detection pipeline."""
    try:
        image        = Image.open(file.stream).convert("RGB")
        original_name = file.filename.rsplit('.', 1)[0]
        return _send_to_api_and_return(image, f"{original_name}_processed.png")

    except Exception as e:
        log.error('Image processing failed: %s', e, exc_info=True)
        return jsonify({'error': 'Image processing failed'}), 500


# --- Internal helpers ---

def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _send_to_api_and_return(image, output_filename):
    """
    Send image to the detection API, apply the returned masks locally,
    and stream the result back to the client.
    """
    tmpfile_path = None
    try:
        # Write to a temp file — it is the shared source for both the multipart
        # API upload and the subsequent local extraction step.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name, "PNG")
            tmpfile_path = f.name

        with open(tmpfile_path, "rb") as f:
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                data=args,
                files={"file": f},
                timeout=API_TIMEOUT,
            )

        if response.status_code != 200:
            log.warning('Detection API returned %d', response.status_code)
            return jsonify({'error': f'Detection API returned status {response.status_code}'}), response.status_code

        masks         = response.json().get("images", [])
        original      = Image.open(tmpfile_path).convert("RGB")
        processed     = extract_and_crop_by_mask(original, masks)

        buf = BytesIO()
        processed.save(buf, format='PNG')
        buf.seek(0)

        log.info('Successfully processed %s', output_filename)
        return send_file(buf, mimetype='image/png', as_attachment=True, download_name=output_filename)

    except requests.Timeout:
        log.error('Detection API timed out after %ds', API_TIMEOUT)
        return jsonify({'error': 'Detection API timed out'}), 504

    except Exception as e:
        log.error('Processing failed: %s', e, exc_info=True)
        return jsonify({'error': 'Processing failed'}), 500

    finally:
        if tmpfile_path and os.path.exists(tmpfile_path):
            os.remove(tmpfile_path)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port  = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host='0.0.0.0', port=port)
