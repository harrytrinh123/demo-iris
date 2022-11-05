import pickle
from flask import Flask, request, render_template
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
# add cors
CORS(app)

@app.route("/predict_iris", methods=['POST'])
def predict():
  sepal_length = request.args.get('sepal_length')
  sepal_width = request.args.get('sepal_width')
  petal_length = request.args.get('petal_length')
  petal_width = request.args.get('petal_width')
  res = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
  return res

@app.route('/')
def home():
  return render_template('index.html')

filename = "iris_bayes.sav"
loaded_model = pickle.load(open(filename, 'rb'))
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
  df_test = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])
  pred = loaded_model.predict(df_test)[0]
  return {"result" : pred}

if __name__ == "__main__":
    app.run(debug=True)
    print(predict_iris(1, 2, 3, 4))
    print('hellow world')