from flask import Flask, request, jsonify
from predict import Predictor

app = Flask(__name__)
predictor = Predictor()  

@app.route('/predict', methods=['POST'])
def predict_spam():
    email_content = request.json['email_content']
    email = {"words": email_content.split()}
    is_spam = predictor.predict(email) 
    return jsonify({"is_spam": is_spam})

if __name__ == '__main__':
    app.run(debug=True)