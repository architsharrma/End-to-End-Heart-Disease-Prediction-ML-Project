
from distutils.log import debug
from fileinput import filename
from flask import Flask, render_template,request
import pickle
import numpy as np

# Load the model:
filename = 'heart-disease-prediction-randomforest-model.pkl'
model = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        chest_pain_type = int(request.form['chest_pain_type'])
        resting_blood_pressure = int(request.form['resting_blood_pressure'])
        fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
        
        resting_ecg = int(request.form['resting_ecg'])
        exercise_angina	 = int(request.form['exercise_angina'])
        oldpeak = float(request.form['oldpeak'])
        ST_slope = int(request.form['ST_slope'])

        data = np.array([[age,sex,chest_pain_type,resting_blood_pressure,fasting_blood_sugar,resting_ecg,exercise_angina,oldpeak,ST_slope]])
        my_prediction = model.predict(data)

        return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)