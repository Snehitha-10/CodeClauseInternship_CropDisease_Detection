from flask import Flask, render_template, request,redirect, url_for
import pickle
import numpy as np
import os
from io import BytesIO
from PIL import Image


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open('iri.pkl','rb'))

def read_file_as_image(data) -> np.ndarray :
    image = np.array(Image.open(BytesIO(data))) # read bytes as image
    img_batch = np.expand_dims(image, 0) # adding 1 more dim
    
    return img_batch
    

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict',methods=['GET','POST'])

def predict():    
    if 'file_input' in request.files:
        
        image = read_file_as_image(request.files['file_input'])
        pred  = model.predict(image)
        
        print('hiii',pred)
        
        pred='yes'
        return render_template('result.html', data=pred)
        
    else:
        pred = 'NO'
        return render_template('result.html', data=pred)   
    
if __name__ == "__main__":
    app.run(debug=True)