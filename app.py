import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the model you saved
MODEL = tf.keras.models.load_model('tomato_model_final.h5', compile=False)

# List of disease names
CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites', 'Target_spot', 'Mosaic_virus', 'Yellow_Leaf_Curl'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((224, 224))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = MODEL.predict(img_array)
    idx = np.argmax(predictions[0])
    confidence = float(predictions[0][idx] * 100)
    
    return jsonify({
        'disease': CLASS_NAMES[idx].replace('_', ' '),
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)