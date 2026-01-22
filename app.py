from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            overallqual = float(request.form["OverallQual"])
            grlivarea = float(request.form["GrLivArea"])
            totalbsmtsf = float(request.form["TotalBsmtSF"])
            garagecars = float(request.form["GarageCars"])
            fullbath = float(request.form["FullBath"])
            yearbuilt = float(request.form["YearBuilt"])

            features = np.array([[overallqual, grlivarea, totalbsmtsf,
                                  garagecars, fullbath, yearbuilt]])

            prediction = model.predict(features)[0]

            prediction = f"{prediction:,.2f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
