from flask import Flask, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots
import matplotlib.pyplot as plt
import io

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

    # Get model predictions
    predictions = model.predict(X)

    # Plotting
    plt.figure()
    plt.scatter(df['X'], df['Y'], color='blue', label='Original Data')
    plt.plot(df['X'], predictions, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
