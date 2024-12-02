from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

model = pickle.load(open("data\Credit_Card_Fraud.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()
        # print("Received data:", data)
        features = data['features']
        feature_values = [features[key] for key in features] 

        prediction = model.predict([feature_values])
        # print("Prediction: ", prediction)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error" : str(e)})

if __name__ == '__main__':
    app.run(debug=True)