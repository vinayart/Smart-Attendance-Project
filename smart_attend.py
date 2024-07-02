from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)

# Load the pre-trained Keras model
loaded_model = load_model(r"C:\python\projects\leaders\new_leaders_model.h5")

# Function to detect faces in an image using Haar Cascade
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    face_images = []
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))  # Resize the detected face image
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
        face_images.append((x, y, w, h, face_img))
    
    return img, face_images

# Define class names
class_names = {
    0: 'Fumio Kishida',
    1: 'Joe Biden',
    2: 'Narendra Modi',
    3: 'Scott Morrison'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    image = None
    predictions = []

    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            original_image, detected_faces = detect_faces(file_path)

            # Perform prediction for each detected face
            for (x, y, w, h, face) in detected_faces:
                face = np.expand_dims(face, axis=0)
                predictions.append(class_names.get(np.argmax(loaded_model.predict(face)), 'Unknown'))

            # Encode the image to base64 for HTML display
            _, buffer = cv2.imencode('.jpeg', original_image)
            image = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html', image=image, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
