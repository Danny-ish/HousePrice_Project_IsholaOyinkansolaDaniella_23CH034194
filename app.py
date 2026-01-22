from flask import Flask, render_template, request
import numpy as np
import joblib
import os  # needed for deployment environment

app = Flask(__name__)

# -------------------------------
# Load the trained model
# -------------------------------
model = joblib.load("model/house_price_model.pkl")

# -------------------------------
# Home route
# -------------------------------
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

            # Make prediction
            pred_value = model.predict(features)[0]

            # Format nicely with commas and 2 decimals
            prediction = f"${pred_value:,.2f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

# -------------------------------
# Deployment-ready settings
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port or default 5000
    app.run(host="0.0.0.0", port=port, debug=True)
