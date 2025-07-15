from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

##import ridge regressor and standard scaler pickle
ridge_modell= pickle.load(open('ridge.pkl', 'rb'))  # Ensure correct model file
standard_scaler= pickle.load(open('scaler.pkl', 'rb'))  # Ensure correct scaler file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
       Temperature=float(request.form.get('Temperature'))
       RH=float(request.form.get('RH'))
       Ws=float(request.form.get('Ws'))
       Rain=float(request.form.get('Rain'))
       FFMC=float(request.form.get('FFMC'))
       DMC=float(request.form.get('DMC'))
       ISI=float(request.form.get('ISI'))
       Classes=float(request.form.get('Classes'))
       Region=float(request.form.get('Region'))

       new_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
       result=ridge_modell.predict(new_data)

       return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
    # Ensure the application runs on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
