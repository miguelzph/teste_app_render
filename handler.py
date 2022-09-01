from flask import Flask, request,Response
from rossmann.Rossmann import Rossmann
import pickle
import pandas as pd
import os

#

# loading model
model = pickle.load(open('model/model_rossmann_retrained.pkl', 'rb'))

# api
app = Flask(__name__)

@app.route('/')
def test_status():
    return {'status': 'ok'}

@app.route('/rossmann/predict', methods=['POST'])
def rossman_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict): # unique line
            test_raw = pd.DataFrame(test_json, index=[0])
        else: # multiple lines
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # Instantiate Rossmanclass
        pipeline = Rossmann()
        
    
        # data cleaning
        df_1 = pipeline.data_cleaning(test_raw.copy())
        # feature engineering
        df_2 = pipeline.feature_engineering(df_1)
        # data prepatarion
        df_3 = pipeline.data_prepatarion(df_2)
        # predicition
        df_response = pipeline.get_prediction(model, test_raw, df_3)
        
        return df_response
        
    else:
        return Response('{No Data}', status=200, mimetype='application/json')
    
if __name__=='__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)