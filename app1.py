# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabetes_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('diabetes.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # if request.method == 'POST':
    #     Pregnancies = int(request.form['pregnancies'])
    #     Glucose = int(request.form['glucose'])
    #     BloodPressure = int(request.form['bloodpressure'])
    #     SkinThickness = float(request.form['skinthickness'])
    #     Insulin = int(request.form['insulin'])
    #     BMI = float(request.form['bmi'])
    #     DiabetesPedigreeFunction = float(request.form['dpf'])
    #     Age = int(request.form['age'])
        
    #     data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    my_prediction = classifier.predict(final_features)


        
    return render_template('diabres.html', prediction_text=my_prediction)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = classifier.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
	app.run(debug=True)