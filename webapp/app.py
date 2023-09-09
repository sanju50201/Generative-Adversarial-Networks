from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the decision tree model
with open('Dtc.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/admin', methods=['POST'])
def admin():
    # Here, we'll simply redirect to the admin page without authenticating for simplicity
    return render_template('admin.html')

current_row_index = 0

import random

@app.route('/get_data')
def get_data():
    global current_row_index

    df = pd.read_csv('cleaned_data_optimized.csv')
    df = df.round(5)

    if current_row_index >= len(df):
        return "No more transactions available"

    row = df.iloc[current_row_index]
    
    
    if random.random() < 0.30:
        result = 'Fraud'
    else:
        result = 'Not Fraud'

    rows_str = f'<tr><td>{row["amount"]}</td><td>{row["oldbalanceOrg"]}</td><td>{row["newbalanceOrig"]}</td><<td>{result}</td></tr>'
    
    current_row_index += 1
    return rows_str

if __name__ == '__main__':
    app.run(debug=True, port=4001)

