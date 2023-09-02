# <YOUR_IMPORTS>
import random

import dill
import pandas as pd
import json
import os
import glob

path = os.environ.get('PROJECT_PATH', '.')

def predict():

    def read_from_json(path_l):

        with open(path_l, 'r') as fl:
            data = dict(json.load(fl))

        for i in data.keys():
            data[i] = [data[i]]

        df = pd.DataFrame.from_dict(data)
        return df

    list_of_files = glob.glob(f'{path}/data/models/*.pkl')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)

    with open(latest_file, 'rb') as fl:
        model = dill.load(fl)
    list_test = os.listdir(f'{path}/data/test/')

    print(list_test)
    for i in list_test:

        df = read_from_json(f'{path}/data/test/{i}' )
        res = model.predict(df)
        df.to_csv(f'{path}/data/predictions/test{i}.csv')

if __name__ == '__main__':
    predict()