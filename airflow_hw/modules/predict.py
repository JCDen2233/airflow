import os
import json
import glob
import logging
import pandas as pd
from datetime import datetime
from os import listdir

import dill

path = os.environ.get('PROJECT_PATH', '..')


def predict():
    result = []    
    model = load_latest_model()
    if(model == 0):
        logging.error('No model file found')
        return
    test_files = [f for f in os.listdir(f'{path}/data/test/') if os.path.isfile(os.path.join(f'{path}/data/test/', f))]
    for filename in test_files:
        result.append([filename.split('.')[0], predict_test_file(f'{path}/data/test/' + filename, model)[0]])
    df = pd.DataFrame(result, columns=['id', 'result'])
    df.to_csv(f'{path}/data/predictions/predicts_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    pass


def load_latest_model():
    try:
        model_list = list(filter(os.path.isfile, glob.glob(f'{path}/data/models/' + "*")))
        model_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)    
        with open(model_list[0],'rb') as file:
    	    model = dill.load(file)
        return model
    except:
        return 0


def predict_test_file(filename, model):
    with open(filename, 'r') as pred_file:
         line = json.load(pred_file)
    df = pd.DataFrame.from_dict([line], orient='columns')
    y = model.predict(df)
    return y


if __name__ == '__main__':
    predict()
