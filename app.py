from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from utils.model_utils import load_model, predict_verse
from utils.image_utils import process_image
import torch

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model at startup
model, label_encoder = load_model('models/best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process image and make prediction
            image_tensor = process_image(filepath)
            prediction, confidence = predict_verse(model, label_encoder, image_tensor, device)
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': f"{confidence:.2f}%",
                'image_url': f"uploads/{filename}"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)