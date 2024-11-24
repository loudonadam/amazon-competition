import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file,'rb') as f_in:
    (dv,model) = pickle.load(f_in)

app = Flask('amazon_check')

@app.route('/predict',methods=['POST'])

def predict():
    category = request.get_json()
    X_category = dv.transform(category)
    X = xgb.DMatrix(X_category, feature_names=dv.get_feature_names_out().tolist())
    y_pred = model.predict(X)[0]
    amazon_in = y_pred > 0.72

    result = {
        'amazon_probability': float(y_pred),
        'amazon_guess': bool(amazon_in)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)