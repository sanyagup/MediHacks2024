from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

@app.route('/api/linear-regression', methods=['GET'])
def linear_regression():
    # Sample data
    data = {
        'X': [1, 2, 3, 4, 5],
        'Y': [2, 4, 5, 4, 5]
    }
    df = pd.DataFrame(data)

    # Linear regression model
    X = df[['X']]
    y = df['Y']
    model = LinearRegression()
    model.fit(X, y)

    # Get model parameters
    intercept = model.intercept_
    coef = model.coef_[0]

    result = {
        'intercept': intercept,
        'coefficient': coef
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
