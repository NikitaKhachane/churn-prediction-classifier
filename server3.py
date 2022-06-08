from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
# load the pickel model
model = pickle.load(open("model_Churn.pkl", "rb"))

@app.route('/')
def home():
   return render_template('index2.html')


@app.route('/predict',methods=['POST','GET'])
def predict():

    features = [float(x) for x in request.form.values()]
    
    final_features = np.array([features])
    prediction = model.predict(final_features) 
 
    output = round(prediction[0], 2)
    if output == 1:
        out = "The Customer is Churned"
    else:
        out = "The Customer is Stayed"
    return render_template("index2.html", prediction_text=out)
    #return render_template('index2.html', prediction_text='The House Category is {}'.format(output))



if __name__ == '__main__':
    app.run(debug=True)
    
    
    