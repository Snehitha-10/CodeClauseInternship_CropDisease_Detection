from flask import Flask, render_template, request, send_from_directory
import pickle
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load your machine learning model here
model = pickle.load(open('iri.pkl', 'rb'))
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a function to process the uploaded image
def process_image(data):
    image = np.array(Image.open(BytesIO(data)))  # Read bytes as an image
    img_batch = np.expand_dims(image, 0)  # Adding 1 more dimension
    return img_batch

# Define a route to display the upload page
@app.route('/')
def upload_page():
    return render_template('upload.html')

# Define a route to handle the uploaded file and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file_input' in request.files:
        image = process_image(request.files['file_input'].read())
        # Make predictions with your model here
        # Replace the following line with your model prediction logic
        prediction = model.predict(image)
        CLASS_NAMES = ['Early Blight', "Late Blight", "Healthy"]

        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        prediction = {'class': predicted_class, 'confidence': confidence}

        uploaded_file = request.files['file_input']
        file_path = "temp_folder/" + uploaded_file.filename  # Define your desired path
        uploaded_file.save(file_path)

        print('Prediction:', prediction)
        print('File Path:', file_path)

        return render_template('result.html', prediction=prediction, filename=file_path)
    else:
        return "No file uploaded."

# Define a route to serve image files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
