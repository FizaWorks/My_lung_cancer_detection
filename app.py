from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("cancer_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    yellow_fingers = int(request.form['yellow_fingers'])
    anxiety = int(request.form['anxiety'])
    chest_pain = int(request.form['chest_pain'])
    coughing = int(request.form['coughing'])
    fatigue = int(request.form['fatigue'])

    features = np.array([[age, smoking, yellow_fingers, anxiety,
                          chest_pain, coughing, fatigue]])

    prediction = model.predict(features)[0]
    result = "High Risk" if prediction == 1 else "Low Risk"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)