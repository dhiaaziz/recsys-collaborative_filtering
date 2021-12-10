#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 22:36:33 2021

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
folder_dir = path + '/outputs'
sim_dir = folder_dir + '/single_similarity.csv'

# path = os.getcwd()
folder_dir = path + '/outputs/'
train_file = folder_dir + 'train_set.csv'
# print(train_file)
data_rating_df = pd.read_csv(train_file)

distance_df = pd.read_csv(sim_dir)

N = 30 #Neighbor count

filt = (distance_df['rank'] <= 30)
distance_df_filtered = distance_df[filt]
distance_df_filtered = distance_df_filtered.loc[:, ~distance_df_filtered.columns.str.contains('^Unnamed')]


user_ids = sorted(set(data_rating_df.user_id))
product_ids = data_rating_df['product_id'].unique()

neighbor_rating_collection = []


for user_id in user_ids:
    print(f'populating rating collection-{user_id}')
    filt = (distance_df['user_id'] == user_id)
    user_df = distance_df_filtered.loc[filt]
    neighbor_ids = set(user_df['user_neighbor'])
    for neighbor_id in neighbor_ids:
        # print(f'neighbor_id-{neighbor_id}')
        for product_id in product_ids:
           
            filt1 = (data_rating_df['user_id'] == neighbor_id) & (data_rating_df['product_id'] == product_id)
            temp_df = data_rating_df.loc[filt1]
            
            filt2 = (distance_df['user_id'] == user_id) & (distance_df['user_neighbor'] == neighbor_id)
            temp_df2 = distance_df.loc[filt2]
            rank_data = temp_df2['rank'].item()
            if(len(temp_df) == 0):
                rating_data = 0
            else: 
                rating_data = temp_df['review_rating'].item()
            
            # continue
            neighbor_rating_data = {'user_id': user_id, 'user_neighbor_id': neighbor_id,'rank': rank_data, 'product_id': product_id, 'neighbor_rating': rating_data}
            neighbor_rating_collection.append(neighbor_rating_data)
            
neighbor_rating_collection_df = pd.DataFrame(neighbor_rating_collection)
neighbor_rating_collection_df.to_csv()
        
# filt1 = (data_rating_df['user_id'] == 21) & (data_rating_df['product_id'] == 610)
# df = data_rating_df.loc[filt1]
# filt2 = 
# df = df.loc[filt2]
print("--- %s seconds ---" % (time.time() - start_time))