#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 03:13:51 2021

@author: dhiaaziz
"""


import pandas as pd
import numpy as np
from numpy import int64
import statistics
import os
import time
import math

start_time = time.time()
path = os.getcwd()
folder_dir = path + '/outputs/'
train_file = folder_dir + 'train_set_multi.csv'
print(train_file)
data_rating_df = pd.read_csv(train_file)

test_file = folder_dir + 'test_set_multi.csv'
test_set_df = pd.read_csv(test_file) 

# 
#     temp_df = test_set_df.iloc[i]
#     user_id = temp_df['user_id']
#     product_id = temp_df['product_id']
#     review_rating = temp_df['review_rating']

sim_dir = folder_dir + '/multi_similarity.csv'
distance_df = pd.read_csv(sim_dir)

N = 80 #Neighbor count
filt = (distance_df['rank'] <= N)
distance_df_filtered = distance_df[filt]
distance_df_filtered = distance_df_filtered.loc[:, ~distance_df_filtered.columns.str.contains('^Unnamed')]

data_rating_df['review_rebuy'].replace(['yes','maybe', 'no'],[5, 3, 1],inplace=True)
data_rating_df['review_valueformoney'].replace(['expensive','just right','cheap'],[1, 3, 5],inplace=True)
data_rating_df['review_packaging'] = data_rating_df['review_packaging'] / 2

test_set_df['review_rebuy'].replace(['yes','maybe', 'no'],[5, 3, 1],inplace=True)
test_set_df['review_valueformoney'].replace(['expensive','just right','cheap'],[1, 3, 5],inplace=True)
test_set_df['review_packaging'] = data_rating_df['review_packaging'] / 2

list_predicted_rating = []
for i in test_set_df.index:
    print(f'predict_rating test set ke-{i}')
    temp_df = test_set_df.iloc[i]
    user_id = temp_df['user_id']
    product_id = temp_df['product_id']
    # review_rating = temp_df['review_rating']
    
    
    
    filt = distance_df_filtered['user_id'] == user_id
    current_df = distance_df_filtered.loc[filt]
    user_neighbor_ids = current_df['user_neighbor'].values
    list_neighbor_rating = []
    for user_neighbor_id in user_neighbor_ids:
        filt1 = (data_rating_df['user_id'] == user_neighbor_id) & (data_rating_df['product_id'] == product_id)
        temp_neighbor_df = data_rating_df.loc[filt1]
        # neighbor_rating = 
        if len(temp_neighbor_df) == 0:
            neighbor_rating = 0
        else:
            rating1 = temp_neighbor_df['review_rebuy'].values[0]
            rating2 = temp_neighbor_df['review_packaging'].values[0]
            rating3 = temp_neighbor_df['review_valueformoney'].values[0]
            neighbor_rating = (rating1 + rating2 + rating3) / 3
            # rating
        list_neighbor_rating.append(neighbor_rating)
        predicted_rating = sum(list_neighbor_rating)/len(list_neighbor_rating)
    list_predicted_rating.append(predicted_rating)
test_set_df['predicted_rating'] = list_predicted_rating


list_overall_rating = []
for i in test_set_df.index:
    overall_df = test_set_df.iloc[i]
    overall_rating = (overall_df['review_rebuy'] + overall_df['review_packaging'] + overall_df['review_valueformoney']) / 3
    list_overall_rating.append(overall_rating)
test_set_df['overall_rating'] = list_overall_rating

list_error = []
for i in test_set_df.index:
    mae_df = test_set_df.iloc[i]
    
    error = mae_df['overall_rating'] - mae_df['predicted_rating']
    error = abs(error)
    list_error.append(error)

mae = sum(list_error) / len(list_error)

print("--- %s seconds ---" % (time.time() - start_time))