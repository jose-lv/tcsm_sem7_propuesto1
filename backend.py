import io
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

# Intérprete TFLite ligero (Render). Si no está instalado (por ejemplo en tu PC
# con TensorFlow), se usa el de TensorFlow como alternativa.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

app = Flask(__name__)
CORS(app)

# El modelo debe estar en la misma carpeta que backend.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'brain_tumor_cnn.tflite')

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tamaño esperado por el modelo: (ancho, alto)
_, IMG_H, IMG_W, _ = input_details[0]['shape']

# Orden alfabético, igual que train_gen.class_indices en Colab
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


def predict_with_tflite(img_array):
    img_array = img_array.astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'brain_tumor_cnn.tflite'})


@app.route('/api/clasificar', methods=['POST'])
def clasificar_api():
    file = request.files.get('image')

    if not file:
        return jsonify({'error': 'No se envió imagen'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'El archivo no es una imagen válida'}), 400

    img = img.resize((int(IMG_W), int(IMG_H)))
    img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    prediction = predict_with_tflite(img_array)
    predicted_class = class_names[int(np.argmax(prediction))]

    probabilities = {
        class_names[i]: float(f"{prob:.4f}")
        for i, prob in enumerate(prediction)
    }

    return jsonify({
        'prediction': f'Predicción: {predicted_class.upper()}',
        'image_name': secure_filename(file.filename or '') or 'imagen.jpg',
        'probs': probabilities
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)