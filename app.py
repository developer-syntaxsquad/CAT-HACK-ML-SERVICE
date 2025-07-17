from flask import Flask, jsonify, request
from flask_cors import CORS
from model import train_model, predict_average_time



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "hello from ml server"}), 200

@app.route("/predict", methods=['POST'])
def predict():
    if(request.is_json):
        data = request.get_json()
        response = predict_average_time(data["machine_id"], data["operator_id"])
        return jsonify({"time" : response})
    else:
        return jsonify({"message" : "Please Pass JSON Data"})



if __name__ == "__main__":
    #driver code
    app.run(host="0.0.0.0", port="8080", debug=True)
    train_model()