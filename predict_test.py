#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

category = {
    "item_count": 2301,
    "vol_purchase_total": 44607831.5,
    "high_rating_perc": .29,
    "rating_mean": 3.15,
    "rating_std": 1.23,
    "best_seller_count": 32
}

response = requests.post(url,json=category).json()

print (response)

if response['amazon_probability']==True:
    print('Amazon is likely in this market.')
else:
    print('This market is probably Amazon-free.')
# In[ ]:




