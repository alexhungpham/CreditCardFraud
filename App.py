from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np




app = Flask(__name__)
model = load_model('cc_pycaret_deploy_1')
Vdata = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']
cols = ['Time',Vdata ,'Amount','Class']


@app.route('/')
def home ():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final =  np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expect Billing will be {}'.format(prediction))

# Doan code nay bi thieu so voi huong dan
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
