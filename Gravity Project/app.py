from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
# make sure to run grav.py 
# I added some error handling here to be fancy, for the most part I think the only error that can occur is if the model file does not exist or can't be loaded.
# most of the following code is outlined or templated from online tutorials/documentation (they are in my resources section).
# I also spent a lot of time making indentations as simple as possible with the bootstap templates when I finished because I don't like the jarring look of the prefered formats. 
app = Flask(__name__)
try:
    model = joblib.load('gravitational_wave_model.pkl')# model loading
    model_loaded = True
except:
    model_loaded = False
    print("You have to run the grav.py file first")  #if the model doesn't exist, this will give an error message
@app.route('/') #page route definitions
def home(): 
    return render_template('home.html')#home page
@app.route('/resume')
def resume():
    return render_template('resume.html') #resume page
@app.route('/projects')
def projects():
    return render_template('projects.html') #projects page
@app.route('/gravitational-waves')
def gravitational_waves():
    return render_template('gravitational_waves.html', model_loaded=model_loaded) #gravitational waves project page. If there is no model, html will not render the box
@app.route('/predict', methods=['POST']) #api endpoint
def predict():
    if not model_loaded:
        return jsonify({'error'}), 500
    try:
        time_sec = float(request.form['time_sec'])
        rh_plus = float(request.form['rh_plus'])
        input_data = np.array([[time_sec, rh_plus]])
        prediction = model.predict(input_data)[0] # makes the prediction
        return jsonify({'prediction': round(prediction, 6), 'input_time': time_sec, 'input_rh_plus': rh_plus, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) 
    except ValueError as e:
        return jsonify({'error'}), 400
    except Exception as e:
        return jsonify({'error'}), 500
if __name__ == '__main__':
    app.run(debug = True)
#Flask Web Framework, Grinberg, M., 2018, O'Reilly Media , Real Python: "Build a JavaScript Front End for a Flask API", 2023, realpython.com, Flask Documentation: "JavaScript, fetch, and JSON", flask.palletsprojects.com, Bootstrap Documentation, 2021, getbootstrap.com