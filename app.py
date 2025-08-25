from flask import Flask, request   # Flask framework
import pickle
from main import generateAI

# Generate or load the model
generateAI()
ai = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "MAVA  Server is running"

@app.route('/predict', methods=['GET'])
def predict():
    temp = request.args.get('temp')
    temp = float(temp)
    data = [[temp]]
    result = ai.predict(data)
    result = result[0]
    return str(result)   # must return string

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)








