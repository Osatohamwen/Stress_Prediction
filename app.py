import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved random forest model
model = joblib.load('RandomForest_model.pkl')

# Define mapping of numeric labels to stress level names
stress_labels = {
    0: "No Stress",
    1: "Mild Stress",
    2: "High Stress"
}

@app.route('/', methods=['GET', 'POST'])
def predict_stress():
    if request.method == 'POST':
        try:
            # Extract input features from form
            features = [float(request.form.get(feature)) for feature in [
                'anxiety_level', 'mental_health_history', 'depression', 'headache', 
                'sleep_quality', 'breathing_problem', 'living_conditions', 
                'academic_performance', 'study_load', 'future_career_concerns', 
                'extracurricular_activities'
            ]]

            # Convert to numpy array and reshape for prediction
            input_array = np.array(features).reshape(1, -1)

            # Predict the numeric stress level
            pred_num = model.predict(input_array)[0]

            # Map prediction to label
            pred_label = stress_labels.get(pred_num, "Unknown")

            return render_template('result.html', stress_level=pred_label)
        except Exception as e:
            return render_template('error.html', error_message=str(e))

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
