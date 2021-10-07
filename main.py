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
filedir = path+'/data.csv' # user-row
# pd.to_numeric(tableFiltered)
tableFiltered.to_csv(filedir, index=True, sep=';')

tableFilteredTrans = tableFiltered.T
filedir = path+'/data_transposed.csv' # product-row
tableFilteredTrans.to_csv(filedir, index=True, sep=';')

print(filedir)


# ==============================
# Step 1 - Calculate Correlation
#   desc:
#       hitung korelasi antar user menggunakan tableFilteredTrans[0].corr(tableFilteredTrans[1]), looping untuk semua user
# 
#   output:
#       correlation_data (pd.DataFrame) - user_id|user_neighbor_id|correlation_coefficient|sorted_rank
# 
#   sample:
#       user_id|user_neighbor_id|correlation_coefficient|rank
#       user750|user46657|0.541|1
#       user750|user8456|0.537|2
#       user750|user76272|0.511|3
#       user67|user6680|0.619|1
#       user67|user9671|0.597|2
#       user67|user25452|0.563|3
# ==============================



# ==============================
# Step 2 - Populasi Rating Neighbor
#   desc:
#       koleksi rating yang diberikan oleh 30 tetangga terdekat, dengan cara filter correlation_data kolom rank < 30
# 
#   output:
#       neighbor_rating (pd.DataFrame) - user_id|user_neighbor_id|rank|product_id|neighbor_rating
# 
#   sample:
#       user_id|user_neighbor_id|rank|product_id|neighbor_rating
#       user750|user46657|1|product_1|0
#       user750|user8456|2|product_1|1
#       user750|user76272|3|product_1|0
#       user750|user46657|1|product_2|5
#       user750|user8456|2|product_2|4
#       user750|user76272|3|product_2|0
#       user750|user46657|1|product_3|4
#       user750|user8456|2|product_3|4
#       user750|user76272|3|product_3|4
# ==============================



# ==============================
# Step 3 - Hitung Rekomendasi
#   desc:
#       hitung agregat rating dari semua neighbor
# 
#   output:
#       predicted_rating (pd.DataFrame) - user_id|product_id|predicted_score
# 
#   sample:
#       user_id|product_id|predicted_score
#       user750|product_1|0.333
#       user750|product_2|3
#       user750|product_3|4
#       user67|product_1|0.
#       user67|product_2|3.
#       user67|product_3|4.
# ==============================



# ==============================
# Step 4 - Evaluasi Error
#   desc:
#       hitung selisih antara predicted rating dengan actual rating dari user
# 
#   output:
#       mean_absolute_error (float) - 0.0
# 
#   sample:
#       0.140212
# ==============================