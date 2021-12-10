#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:06:51 2021

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
train_file = folder_dir + 'train_set.csv'
print(train_file)
data_rating_df = pd.read_csv(train_file)

test_file = folder_dir + 'test_set.csv'
test_set_df = pd.read_csv(test_file) 

# 
#     temp_df = test_set_df.iloc[i]
#     user_id = temp_df['user_id']
#     product_id = temp_df['product_id']
#     review_rating = temp_df['review_rating']

sim_dir = folder_dir + '/single_similarity.csv'
distance_df = pd.read_csv(sim_dir)

N = 80 #Neighbor count
filt = (distance_df['rank'] <= N)
distance_df_filtered = distance_df[filt]
distance_df_filtered = distance_df_filtered.loc[:, ~distance_df_filtered.columns.str.contains('^Unnamed')]

list_predicted_rating = []
for i in test_set_df.index:
    print(f'predict_rating test set ke-{i}')
    temp_df = test_set_df.iloc[i]
    user_id = temp_df['user_id']
    product_id = temp_df['product_id']
    review_rating = temp_df['review_rating']
    
    
    
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
            neighbor_rating = temp_neighbor_df['review_rating'].values[0]
        list_neighbor_rating.append(neighbor_rating)
        predicted_rating = sum(list_neighbor_rating)/len(list_neighbor_rating)
    list_predicted_rating.append(predicted_rating)
test_set_df['predicted_rating'] = list_predicted_rating

list_error = []
for i in test_set_df.index:
    mae_df = test_set_df.iloc[i]
    error = mae_df['review_rating'] - mae_df['predicted_rating']
    error = abs(error)
    list_error.append(error)

mae = sum(list_error) / len(list_error)
    




