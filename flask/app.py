from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained XGBoost model, TF-IDF Vectorizer, and Label Encoder
xgb_model = joblib.load('D:/smart, project-3/flask/models/sentiment_analysis_xgb_model.pkl')
vectorizer = joblib.load('D:/smart, project-3/flask/models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('D:/smart, project-3/flask/models/label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def index1():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['statement']
        data = [user_input]
        vectorized_input = vectorizer.transform(data)
        prediction = xgb_model.predict(vectorized_input)
        
        # Decode the prediction
        decoded_prediction = label_encoder.inverse_transform(prediction)
        
        # Return the result as a JSON response
        return jsonify({'prediction': decoded_prediction[0]})
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
