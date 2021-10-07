#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:24:16 2021

@author: dhiaaziz
"""

import pandas as pd
import numpy as np
from numpy import int64
import os

# import json

ratings_df = pd.read_json("./top10cat_balanced.json", lines=True)
# ratings_df = pd.DataFrame("./data_fix.json", lines=False)

ratings_df.head(n=10)

ratings_df_filtered = ratings_df.copy(deep=True)
ratings_df_filtered = ratings_df_filtered[['review_id','product_id','product_name','user_id','review_rating','review_packaging','review_rebuy','review_valueformoney']]
ratings_df_filtered['review_rebuy'].replace(['yes','maybe', 'no'],['5','3','1'],inplace=True)
ratings_df_filtered['review_valueformoney'].replace(['expensive','just right','cheap'],['1','3','5'],inplace=True)

# display(ratings_df_filtered)


# =============================================================================
# for key,value in ratings_df_filtered.items():
#     print (key)
#     print(value)
# =============================================================================

    
# print(type(ratings_df_filtered))
user_ids = set(ratings_df_filtered.user_id)
product_ids = set(ratings_df_filtered.product_id)
review_ids = set(ratings_df_filtered.review_id)


table = pd.pivot_table(ratings_df_filtered, index='user_id', columns='product_id', values='review_rating')
cols_to_int = []


count= table.count(axis=1)
countBool = table.count(axis=1) >= 5
countTrue = countBool.sum()

tableFiltered = table[table.count(axis=1) >= 5]
tableFiltered = tableFiltered.fillna(0)
tableFiltered = tableFiltered.astype(int)


path = os.getcwd()
filedir = path+'/data.csv'
# pd.to_numeric(tableFiltered)
tableFiltered.to_csv(filedir, index=True, sep=';')

tableFilteredTrans = tableFiltered.T
filedir = path+'/data_transposed.csv'
tableFilteredTrans.to_csv(filedir, index=True, sep=';')

print(filedir)