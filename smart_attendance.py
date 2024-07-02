import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# Set the path to the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained Keras model
loaded_model = load_model(r"C:\mini_project_2\smart attendance\famous_personalities_model.h5")

# Function to detect faces in an image using Haar Cascade
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    face_images = []
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (120, 120))  # Resize the detected face image
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel image
        face_images.append((x, y, w, h, face_img))
    
    return face_images

class_names = [
    'anushka','barack','bill gates','dalai lama','indira','milinda','modi','pichai','vikas','virat']

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    

    file = request.files['file']
    print(file.filename)
    predicted_class_names=[]

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the uploaded image
        input_image = cv2.imread(filepath)

        # Detect faces in the input image
        detected_faces = detect_faces(input_image)

        # Perform prediction for each detected face
        for (x, y, w, h, face) in detected_faces:
            # Make predictions using the loaded model
            face = np.expand_dims(face, axis=0)
            predictions = loaded_model.predict(face)
            
            # Get the predicted class (index)
            predicted_class = np.argmax(predictions)
            
            # Retrieve class name from the list or print 'Unknown' if not found
            class_name = class_names[predicted_class]  
            predicted_class_names.append(class_name)
            cv2.rectangle(input_image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Draw rectangle around face
            
            # Draw text label
            cv2.putText(input_image, f"Class: {class_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Encode the annotated image as Base64
        _, buffer = cv2.imencode('.png', input_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        absent=[]
        for names1 in class_names:
            if names1 not in predicted_class_names:
                absent.append(names1)
        # Return the Base64 encoded image directly to the client for display
        return render_template('index.html', image=annotated_image_base64,predictions=predicted_class_names,absents=absent,count=len(predicted_class_names))

if __name__ == '__main__':
    app.run(debug=True)
