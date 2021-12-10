#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 02:24:52 2021

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

data_rating_df['review_rebuy'].replace(['yes','maybe', 'no'],[5, 3, 1],inplace=True)
data_rating_df['review_valueformoney'].replace(['expensive','just right','cheap'],[1, 3, 5],inplace=True)
data_rating_df['review_packaging'] = data_rating_df['review_packaging'] / 2

def euclidean_distance(user1, user2):
    # print('a')     
    filt_user = (data_rating_df['user_id'] == user1)
    user1_df = data_rating_df[filt_user]
    filt_user2 = (data_rating_df['user_id'] == user2)
    user2_df = data_rating_df[filt_user2]
    
    rated_u = user1_df['product_id'].unique()
    rated_k = user2_df['product_id'].unique()
    
    intersection = [value for value in rated_u if value in rated_k]
    if(len(intersection) < 1):
        return -9999
    
    dtotal = 0
    
    for product_id in intersection:
        # print(product_id)
        filt = (data_rating_df['user_id'] == user1) & (data_rating_df['product_id'] == product_id)
        # user_i = 
        values1 = data_rating_df[filt]['review_rebuy'].values[0]
        values2 = data_rating_df[filt]['review_packaging'].values[0]
        values3 = data_rating_df[filt]['review_valueformoney'].values[0]
        
        # values1 = np.array((hitung, 2))
        filt = (data_rating_df['user_id'] == user2) & (data_rating_df['product_id'] == product_id)
        values4 = data_rating_df[filt]['review_rebuy'].values[0]
        values5 = data_rating_df[filt]['review_packaging'].values[0]
        values6 = data_rating_df[filt]['review_valueformoney'].values[0]
        # values2 = np.array((hitung2, 5))
        d = math.sqrt(((values1-values4) ** 2) + ((values2-values5) ** 2) + ((values3-values6) ** 2))
        dtotal += d
        
    distance = (1/len(intersection)) * dtotal
    
    return distance

#train set ids
user_ids_all = sorted(set(data_rating_df.user_id))
product_ids = sorted(set(data_rating_df.product_id))



distance_list = []
for user_u in user_ids_all:
    # if (user_u >= 1000):
    #     break
    
    for user_j in user_ids_all:   
        print(f'Calculating Similarity User-{user_u} to user-{user_j}')
        if user_u == user_j:
            continue
        distance = euclidean_distance(user_u, user_j)
        if(distance == -9999):
            continue
        distance_data = {'user_id': user_u, 'user_neighbor': user_j, "euclid_distance": distance}
        distance_list.append(distance_data)


# test = euclidean_distance(750, 23899)
distance_df = pd.DataFrame(distance_list)
distance_df['rank'] = distance_df.groupby('user_id')['euclid_distance'].rank(ascending=True, method='first')

path = os.getcwd()
folder_dir = path + '/outputs'
file_dir = folder_dir + '/multi_similarity.csv'

distance_df.to_csv(file_dir, sep=",")

print("--- %s seconds ---" % (time.time() - start_time))
