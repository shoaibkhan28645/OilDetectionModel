
import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from models_db import db, User, DetectionHistory
from predict import OilDetectionPredictor
from datetime import datetime

print("Initializing Flask app...")
app = Flask(__name__)

# Configuration
# Configure the upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.curdir), 'uploads')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///oil_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'super-secret-key-change-this-in-production' 

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)

# Create DB tables
with app.app_context():
    db.create_all()

# Load the model
print("Loading model...")
model_path = '../models/oil_detection_transfer_learning_20250901_033505.h5'
# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

try:
    predictor = OilDetectionPredictor(model_path)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

# --- Web Routes (Existing) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_web', methods=['POST'])
def predict_web():
    if 'image' not in request.files:
        return render_template('index.html', error='No image selected')
    image = request.files['image']
    if image.filename == '':
        return render_template('index.html', error='No image selected')
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        if predictor:
            result = predictor.predict_single_image(image_path, show_image=False)
            return render_template('result.html', result=result, image_path=image_path)
        return render_template('index.html', error='Model not loaded')

# --- API Routes (New) ---

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing username or password'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 400
    
    new_user = User(username=data['username'])
    new_user.set_password(data['password'])
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing username or password'}), 400
        
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'message': 'Invalid username or password'}), 401
    
    access_token = create_access_token(identity=str(user.id))
    return jsonify({'access_token': access_token, 'user_id': user.id, 'username': user.username}), 200

@app.route('/api/predict', methods=['POST'])
@jwt_required()
def predict_api():
    current_user_id = get_jwt_identity()
    
    if 'image' not in request.files:
        return jsonify({'message': 'No image provided'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'message': 'No image selected'}), 400
        
    # Generate unique filename to avoid overwrites
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)
    
    print(f"DEBUG: Saved image to {image_path}")
    if not os.path.exists(image_path):
         print("DEBUG: File does not exist after save!")
         return jsonify({'message': 'File save failed'}), 500

    if not predictor:
         return jsonify({'message': 'Model not loaded'}), 500

    try:
        result = predictor.predict_single_image(image_path, show_image=False)
    except Exception as e:
        print(f"DEBUG: Prediction exception: {e}")
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500
        
    if result is None:
        print("DEBUG: Predictor returned None")
        return jsonify({'message': 'Prediction failed (processor returned None)'}), 500

    # Save history
    # Convert numpy types to native python types for DB and JSON
    confidence = float(result['confidence'])
    predicted_class = result['predicted_class']
    
    history_entry = DetectionHistory(
        user_id=int(current_user_id),
        image_path=filename, # Store relative path/filename
        predicted_class=predicted_class,
        confidence=confidence
    )
    db.session.add(history_entry)
    db.session.commit()
    
    return jsonify({
        'result': {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': result['probabilities']
        },
        'image_url': f"/uploads/{filename}",
        'history_id': history_entry.id
    }), 200

@app.route('/api/history', methods=['GET'])
@jwt_required()
def get_history():
    current_user_id = get_jwt_identity()
    if not current_user_id: # Should be handled by jwt_required, but safety check
         return jsonify({'message': 'Unauthorized'}), 401
         
    history = DetectionHistory.query.filter_by(user_id=int(current_user_id)).order_by(DetectionHistory.timestamp.desc()).all()
    
    history_list = []
    for entry in history:
        history_list.append({
            'id': entry.id,
            'image_url': f"/uploads/{entry.image_path}",
            'predicted_class': entry.predicted_class,
            'confidence': entry.confidence,
            'timestamp': entry.timestamp.isoformat()
        })
        
    return jsonify(history_list), 200

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
