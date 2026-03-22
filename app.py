import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load the model
# The file `loan_approval_xgboost_pipeline.pkl` contains a scikit-learn Pipeline
MODEL_PATH = 'loan_approval_xgboost_pipeline.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Render the main frontend HTML."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload with the following keys:
    - person_age (float)
    - person_gender (str) [female, male]
    - person_education (str) [High School, Bachelor, Master, Associate, Doctorate]
    - person_income (float)
    - person_emp_exp (int)
    - person_home_ownership (str) [RENT, OWN, MORTGAGE, OTHER]
    - loan_amnt (float)
    - loan_intent (str) [PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION]
    - loan_int_rate (float)
    - loan_percent_income (float)
    - cb_person_cred_hist_length (float)
    - credit_score (int)
    - previous_loan_defaults_on_file (str) [Yes, No]
    """
    if model is None:
        return jsonify({'error': 'Model not loaded correctly.'}), 500

    try:
        data = request.json
        
        # Construct DataFrame from the JSON
        # Values must be cast to appropriate types or Pandas will infer from the lists
        df = pd.DataFrame([data])
        
        # Model returns a 1D array of predictions
        prediction = model.predict(df)[0]
        
        # 1 = Approved, 0 = Rejected
        loan_status = int(prediction)
        
        return jsonify({
            'loan_status': loan_status,
            'message': 'Approved' if loan_status == 1 else 'Rejected'
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run server locally on port 5000
    app.run(debug=False, port=5000)
