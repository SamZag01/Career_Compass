import json
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
from app import app
from user_data import update_data
# from trainingmodel import calculate_scores, suggestCareers
# import pandas as pd
import os
# app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio= SocketIO(app)

import json
import os


@app.route('/')
def index():
    # Define default scores and careers
    default_scores = {
        'Numerical_Aptitude': 0,
        'Spatial_Aptitude': 0,
        'Perceptual_Aptitude': 0,
        'Abstract_Reasoning': 0,
        'Verbal_Reasoning': 0
    }

    default_careers = []
    # Overwrite scores and careers every time the page is loaded
    with open('static/scores.json', 'w') as file:
        json.dump(default_scores, file)

    with open('static/suggested_careers.json', 'w') as file:
        json.dump(default_careers, file)

    # # Check if the scores.json file exists, initialize or overwrite if necessary
    # if not os.path.exists('static/scores.json') or os.environ.get('RESET_DATA', 'False') == 'True':
    #     with open('static/scores.json', 'w') as file:
    #         json.dump(default_scores, file)
    #
    # # Check if the suggested_careers.json file exists, initialize or overwrite if necessary
    # if not os.path.exists('static/suggested_careers.json') or os.environ.get('RESET_DATA', 'False') == 'True':
    #     with open('static/suggested_careers.json', 'w') as file:
    #         json.dump(default_careers, file)
    #         print("initialization done at view")

    # Load scores from JSON file
    try:
        with open('static/scores.json', 'r') as file:
            scores = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        scores = default_scores

    # Load suggested careers from JSON file
    try:
        with open('static/suggested_careers.json', 'r') as file:
            suggested_careers = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        suggested_careers = default_careers

    return render_template('index.html', scores=scores, suggested_careers=suggested_careers)


# @app.route('/')
# def index():
#     # Load scores from JSON file
#     try:
#         with open('static/scores.json', 'r') as file:
#             scores = json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         scores = {
#             'Numerical_Aptitude': 0,
#             'Spatial_Aptitude': 0,
#             'Perceptual_Aptitude': 0,
#             'Abstract_Reasoning': 0,
#             'Verbal_Reasoning': 0
#         }
#
#     # Load suggested careers from JSON file
#     try:
#         with open('static/suggested_careers.json', 'r') as file:
#             suggested_careers = json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         suggested_careers = []
#
#     return render_template('index.html', scores=scores, suggested_careers=suggested_careers)

@app.route('/saveData', methods=['POST'])
def save_data():
    data = request.get_json()
    with open('data.json', 'w') as file:
        json.dump(data, file)
    update_data()
    print("View data updated")

    file_path = os.path.join('static','scores.json')

    with open(file_path, 'r') as file:
        scores = json.load(file)

    file_path1=os.path.join('static','suggested_careers.json')

    with open(file_path1, 'r') as file:
        suggested_careers = json.load(file)

    # socketio.emit('update_data', {'scores': scores, 'suggested_careers': suggested_careers}, broadcast=True)

    return jsonify({'scores': scores, 'suggested_careers': suggested_careers})

    # return render_template('index.html', scores=scores, suggested_careers=suggested_careers)

    # return 'Data successfully saved at view', 200



# app=Flask(__name__)
# socketio=SocketIO(app)
#
# @socketio.on('connect')
# def handle_connect():
#     emit('update_data', read_json_data(), broadcast=True)
#
# def read_json_data():
#     with open('scores.json') as f:
#         return json.load(f)
#
# @socketio.on('update')
# def handle_update():
#     emit('update_data', read_json_data(), broadcast=True)

# @app.route('/updateData', methods=['GET'])
# def update_data():
#     scores=request.get_json()
#     # Read data from JSON file
#     try:
#         with open('scores.json') as file:
#             scores = json.load(file)
#     except FileNotFoundError:
#         scores = {}
#
#     # Read data from JSON file
#     try:
#         with open('suggested_careers.json') as file:
#             suggested_careers = json.load(file)
#     except FileNotFoundError:
#         suggested_careers = []
#
#     return jsonify({'scores': scores, 'suggested_careers': suggested_careers})
#     # return render_template('index.html', scores=scores, suggested_careers=suggested_careers)

# @app.route('/calculateScores', methods=['POST'])
# def calculate_and_save_scores():
#     data = request.get_json()
#     user_data = pd.DataFrame(data)
#
#     OPN = 'O'
#     CON = 'C'
#     EXT = 'E'
#     AGR = 'A'
#     NES = 'N'
#
#     df_input = calculate_scores(user_data, OPN, CON, EXT, AGR, NES)
#     scores = df_input.to_dict(orient='records')[0]
#
#     with open('scores.json', 'w') as json_file:
#         json.dump(scores, json_file)
#
#     return 'Scores successfully calculated and saved', 200
#
# @app.route('/updateData', methods=['GET'])
# def update_data():
#     try:
#         with open('scores.json', 'r') as file:
#             scores = json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         scores = {
#             'Numerical_Aptitude': 0,
#             'Spatial_Aptitude': 0,
#             'Perceptual_Aptitude': 0,
#             'Abstract_Reasoning': 0,
#             'Verbal_Reasoning': 0
#         }
#
#     try:
#         with open('suggested_careers.json', 'r') as file:
#             suggested_careers = json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         suggested_careers = []
#
#     return jsonify({'scores': scores, 'suggested_careers': suggested_careers})