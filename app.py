# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from models import db, OcrResult
from utils import combine_images_to_pdf, detect_and_extract_face, call_gemini_ocr, convert_pdf_first_page_to_image

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# --- App Configuration ---
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACTED_FACES_FOLDER'] = 'extracted_faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# --- Database Configuration ---
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)

# Create upload directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_FACES_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Helper function for processing logic ---
def process_files(files):
    single_file = files.get('single_file')
    side1_file = files.get('side1_file')
    side2_file = files.get('side2_file')
    
    file_to_process = None
    original_filename = ""
    face_source_image_path = None

    if single_file and single_file.filename != '' and allowed_file(single_file.filename):
        filename = secure_filename(single_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        single_file.save(filepath)
        
        file_to_process = filepath
        original_filename = filename
        
        file_extension = filename.rsplit('.', 1)[1].lower()
        if file_extension in ['jpg', 'jpeg', 'png']:
            face_source_image_path = filepath
        elif file_extension == 'pdf':
            print("PDF detected, attempting to convert first page for face detection...")
            face_source_image_path = convert_pdf_first_page_to_image(
                filepath, 
                app.config['UPLOAD_FOLDER']
            )
            if not face_source_image_path:
                print("Could not extract image from PDF for face detection.")

    elif side1_file and side2_file and allowed_file(side1_file.filename) and allowed_file(side2_file.filename):
        s1_filename = secure_filename(side1_file.filename)
        s2_filename = secure_filename(side2_file.filename)
        s1_filepath = os.path.join(app.config['UPLOAD_FOLDER'], s1_filename)
        s2_filepath = os.path.join(app.config['UPLOAD_FOLDER'], s2_filename)
        side1_file.save(s1_filepath)
        side2_file.save(s2_filepath)

        face_source_image_path = s1_filepath

        pdf_path, pdf_filename = combine_images_to_pdf(s1_filepath, s2_filepath, app.config['UPLOAD_FOLDER'])
        if pdf_path:
            file_to_process = pdf_path
            original_filename = pdf_filename
        else:
            return {"error": "Failed to combine images into PDF"}, 400
    else:
        return {"error": "Invalid file submission. Please provide a single file or two side images."}, 400

    if not file_to_process:
        return {"error": "File to process could not be determined."}, 400

    extracted_data = call_gemini_ocr(file_to_process)
    if "error" in extracted_data:
        return extracted_data, 500

    face_path = None
    if face_source_image_path:
        face_path = detect_and_extract_face(face_source_image_path, app.config['EXTRACTED_FACES_FOLDER'])

    new_result = OcrResult(
        original_filename=original_filename,
        extracted_data=extracted_data,
        face_image_path=face_path
    )
    db.session.add(new_result)
    db.session.commit()
    
    return {"success": True, "data": extracted_data, "face_image": face_path, "id": new_result.id}, 200

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    results = OcrResult.query.order_by(OcrResult.processed_at.desc()).all()
    return render_template('history.html', results=results)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/extracted_faces/<path:filename>')
def serve_face(filename):
    return send_from_directory(app.config['EXTRACTED_FACES_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_from_form():
    if not request.files:
        return "No file part", 400
    
    files = {
        'single_file': request.files.get('single_file'),
        'side1_file': request.files.get('side1_file'),
        'side2_file': request.files.get('side2_file'),
    }
    
    result, status_code = process_files(files)
    
    if status_code == 200:
        return redirect(url_for('history'))
    else:
        error_message = "An unknown error occurred."
        if isinstance(result, dict) and 'error' in result:
            error_message = result['error']
        elif isinstance(result, str):
            error_message = result
        return error_message, status_code

@app.route('/api/ocr', methods=['POST'])
def upload_from_api():
    if not request.files:
        return jsonify({"error": "No file part in the request"}), 400
    files = {
        'single_file': request.files.get('file'),
        'side1_file': request.files.get('side1'),
        'side2_file': request.files.get('side2'),
    }
    result, status_code = process_files(files)
    return jsonify(result), status_code

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')