# <YOUR_IMPORTS>
import random

import dill
import pandas as pd
import json
import os
import glob
import glob

path = os.environ.get('PROJECT_PATH', '.')

def predict(name_model):

    def read_from_json(path_l):

        with open(path_l, 'r') as fl:
            data = dict(json.load(fl))

        for i in data.keys():
            data[i] = [data[i]]

        df = pd.DataFrame.from_dict(data)
        return df


    with open(name_model, 'rb') as fl:
        model = dill.load(fl)
    list_test = os.listdir(f'{path}/data/test/')

    print(list_test)
    for i in list_test:

        df = read_from_json(f'{path}/data/test/{i}' )
        res = model.predict(df)
        df.to_csv(f'{path}/data/predictions/test.csv')

if __name__ == '__main__':
    predict()