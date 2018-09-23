# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/train.csv")
data = data.iloc[:,1:]
data_X = data.iloc[:,0:7]
data_Y = data.iloc[:,7]
data_X["CALL_TYPE"] = data_X["CALL_TYPE"].astype('category')
le = preprocessing.LabelEncoder()
le.fit(data_X["CALL_TYPE"].cat.categories.tolist())
data_X["CALL_TYPE"] = le.transform(data_X["CALL_TYPE"])
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
data_call_type = data_X["CALL_TYPE"].values.reshape(len(data_X["CALL_TYPE"]), 1)
data_call_type = onehot_encoder.fit_transform(data_call_type)
normalize = preprocessing.MinMaxScaler()
data_taxi_id = normalize.fit_transform(data_X["TAXI_ID"].values.reshape(len(data_X["TAXI_ID"]),1))
print(data_taxi_id)