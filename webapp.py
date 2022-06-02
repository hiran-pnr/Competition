# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:25:30 2022

@author: Lenovo
"""

from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cols_when_model_builds = model.get_booster().feature_names
print(cols_when_model_builds)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=["POST"])
def predict():
    education=request.values['education']
    gender=request.values['gender']
    recruitment_channel=request.values['recruitment_channel']
    no_of_trainings=request.values['no_of_trainings']
    age=request.values['age']
    previous_year_rating=request.values['previous_year_rating']
    length_of_service=request.values['length_of_service']
    avg_training_score=request.values['avg_training_score']
    awards_won=request.values['awards_won']
    sample=[education,gender,recruitment_channel,no_of_trainings,age,previous_year_rating,length_of_service,avg_training_score,awards_won]
    clean_sample = [float(i) for i in sample]
    exp = np.array(clean_sample).reshape(1,-1)
    output=model.predict(exp)
    output=output.item()
    if output==1:
        pred_text="Congratz! You are eligible for promotion.."
    else:
        pred_text="Sorry! You are NOT eligible for promotion.."
    return render_template('result.html', prediction_text=pred_text)
if __name__ == '__main__':
	app.run(port=8000)
    

