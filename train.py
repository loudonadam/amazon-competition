#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score, mean_squared_error
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import xgboost as xgb
import seaborn as sns
import pickle

#parameters
output_file = 'model.bin'

xgb_params = {
    'eta': 0.5, 
    'max_depth': 2,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

print('Importing data.')
#data import and cleaning
data = pd.read_csv(r'C:\Users\loudo\ML Zoomcamp\amazon_products.csv')
data = data.dropna()
data = pd.DataFrame(data)
data = data[data['price'] != 0] 

# new measure creation
mask = (data['listPrice']!=0)
mask_valid = [mask]
data['discount'] = 0.0
data.loc[mask,'discount'] = data['listPrice'] - data['price']
data['high_rating'] = (data['stars']>=4.5)
data['vol_purchase'] = data['boughtInLastMonth'] * data['price']
data['perc_discount'] = data['discount']/ data['price']

print('Marking Amazon-owned products.')
#marking amazon owned items
amazon_brands = ['Pinzon', 'Amazon Basics', 'AmazonBasics', 'Solimo', 'Amazon Elements','AmazonElements', 'Amazon Brand','Mama Bear','Wickedly Prime','Whole Foods','AmazonFresh','Vedaka','Goodthreads','206 Collective','Amazon Essentials','AmazonEssentials','Core 10']
pattern = '|'.join(amazon_brands)
data['amazon_owned'] = data['title'].str.contains(pattern,case=False).astype(int)

#aggregating down to the category level and creating final dataframe
df = data.groupby(
    ['category_id']
).agg(
    item_count=('title','count'),
    amazon_owned_total=('amazon_owned', 'sum'),
    vol_purchase_total=('vol_purchase', 'sum'),
    high_rating_perc=('high_rating','mean'),
    rating_mean=('stars','mean'),
    rating_std=('stars','std'),
    best_seller_count=('isBestSeller','sum')
).reset_index()

df['amazon_owned']=(df.amazon_owned_total>0).astype(int)
df.drop('amazon_owned_total',axis=1,inplace=True)
df.set_index('category_id',inplace=True)

#splitting the data into train, test, and val making sure sizes make sense
df_full_train, df_test = train_test_split(df,test_size=.2, random_state=50)

#preserving outcome variable
y_train = (df_full_train.amazon_owned == 1).astype('int').values
y_test = (df_test.amazon_owned == 1).astype('int').values

#removing outcome variable from dfs
del df_full_train['amazon_owned']
del df_test['amazon_owned']


# preparing dicts
train_dicts = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

print('Training the model.')
#training
dfulltrain = xgb.DMatrix(X_train, label=y_train,
                    feature_names=dv.get_feature_names_out().tolist())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out().tolist())
model = xgb.train(xgb_params, dfulltrain, num_boost_round=100)
y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)

print('AUC: ', auc)

#save the model
with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)
