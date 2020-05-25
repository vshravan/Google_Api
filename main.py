import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('flower-v1.pkl', 'rb'))

# final_features = np.array([1,0.1,0,1]).reshape(1,-1)
# print(final_features)
# prediction = model.predict(final_features)
# print(prediction)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x*2) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,-1)

    #print(final_features)
    prediction = model.predict(final_features)


    return render_template('index.html', prediction_text='Predicted Flower {}'.format(prediction))


if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=8000)
    app.run(debug=True)
