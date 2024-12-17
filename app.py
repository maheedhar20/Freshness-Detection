from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('rottenvsfresh.h5')

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(150), nullable=False)
    freshness_index = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.image_name} - {self.freshness_index}>"

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            # Open and preprocess the image
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((100, 100))  # Match the model's expected input size
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(img_array)
            freshness_index = float(predictions[0][0])  # Assuming a single output value between 0 and 1

            # Save prediction to the database
            new_prediction = Prediction(image_name=file.filename, freshness_index=freshness_index)
            db.session.add(new_prediction)
            db.session.commit()

            # Return the freshness index as a response
            return jsonify({'freshness_index': freshness_index})
        
        return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        import traceback
        print("Error occurred:", str(e))
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

# Route to view prediction history
@app.route('/history')
def history():
    all_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=all_predictions)

if __name__ == '__main__':
    # Ensure the database is created
    with app.app_context():
        db.create_all()
    app.run(debug=True)
