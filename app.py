# -*- coding: utf-8 -*-

import numpy as np
import os
from flask import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model

model = load_model("keras_model.h5", compile=False)

# Create application
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    result=["Left Paralysis","Right Paralysis","Normal"]
    file = request.files['file']
    file_path =file.filename
    file.save(file_path)
    img = load_img(file_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    prediction = list(model.predict(x)[0])
    
    label=result[prediction.index(max(prediction))]
    return jsonify({"result":label})

if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host="0.0.0.0", port=port)
