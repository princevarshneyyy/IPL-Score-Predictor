from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# Load models
score_model = None
win_model = None

try:
    with open('models/score_model.pkl', 'rb') as f:
        score_model = pickle.load(f)
    with open('models/win_model.pkl', 'rb') as f:
        win_model = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}. Please run train_models.py first.")

teams = ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'SRH', 'PBKS', 'DC']
team_mapping = {team: i for i, team in enumerate(teams)}

@app.route('/api/predict-score', methods=['POST'])
def predict_score():
    try:
        data = request.json
        batting_team = team_mapping.get(data.get('batting_team'), 0)
        bowling_team = team_mapping.get(data.get('bowling_team'), 1)
        overs = float(data.get('overs', 0))
        current_runs = int(data.get('current_runs', 0))
        wickets = int(data.get('wickets', 0))
        runs_last_5 = int(data.get('runs_last_5', 0))
        wickets_last_5 = int(data.get('wickets_last_5', 0))
        
        features = np.array([[batting_team, bowling_team, overs, current_runs, wickets, runs_last_5, wickets_last_5]])
        
        predicted_score = int(score_model.predict(features)[0])
        # Simple confidence interval logic for UI purposes
        lower_bound = predicted_score - np.random.randint(4, 8)
        upper_bound = predicted_score + np.random.randint(4, 8)
        
        return jsonify({
            'success': True,
            'predicted_score': predicted_score,
            'score_range': f"{lower_bound} to {upper_bound}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict-win', methods=['POST'])
def predict_win():
    try:
        data = request.json
        batting_team = team_mapping.get(data.get('batting_team'), 0)
        bowling_team = team_mapping.get(data.get('bowling_team'), 1)
        overs = float(data.get('overs', 0))
        current_runs = int(data.get('current_runs', 0))
        wickets = int(data.get('wickets', 0))
        target_score = int(data.get('target_score', 160)) # Using target_score instead of prediction for simple win model
        
        features = np.array([[batting_team, bowling_team, overs, current_runs, wickets, target_score]])
        
        win_prob_class_1 = win_model.predict_proba(features)[0][1]
        
        batting_win_prob = round(win_prob_class_1 * 100, 1)
        bowling_win_prob = round((1 - win_prob_class_1) * 100, 1)
        
        return jsonify({
            'success': True,
            'batting_team_win_probability': batting_win_prob,
            'bowling_team_win_probability': bowling_win_prob
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
