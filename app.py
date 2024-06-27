from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)

@app.route('/api/linear-regression', methods=['POST'])
def linear_regression():
    # Get the uploaded file and columns from the request
    if 'file' not in request.files:
        return jsonify({"error": "File must be provided"}), 400
    
    file = request.files['file']
    features = request.form.get('features')
    target = request.form.get('target')

    if not file or not features or not target:
        return jsonify({"error": "File, features, and target must be provided"}), 400

    # Convert features string to a list
    features = [x.strip() for x in features.split(',')]

    # Read the CSV file from the uploaded file
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {e}"}), 400

    # Check if required columns exist in the dataset
    required_columns = set(features + [target])
    if not required_columns.issubset(df.columns):
        return jsonify({"error": f"CSV file must contain the following columns: {', '.join(required_columns)}"}), 400

    # Linear regression model
    X = df[features]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)

    # Get model predictions
    predictions = model.predict(X)

    # Plotting
    plt.figure()
    for feature in features:
        plt.scatter(df[feature], df[target], label=f'{feature} vs {target}')
        plt.plot(df[feature], predictions, linewidth=2, label=f'Regression Line ({feature})')
    
    plt.xlabel('Features')
    plt.ylabel(target)
    plt.legend()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
