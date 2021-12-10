#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:24:46 2021

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

#train set ids
user_ids_all = sorted(set(data_rating_df.user_id))
product_ids = sorted(set(data_rating_df.product_id))


user1 = 1270
user2 = 1883
# print('a')     
filt_user = (data_rating_df['user_id'] == user1)
user1_df = data_rating_df[filt_user]
filt_user2 = (data_rating_df['user_id'] == user2)
user2_df = data_rating_df[filt_user2]

rated_u = user1_df['product_id'].unique()
rated_k = user2_df['product_id'].unique()

intersection = [value for value in rated_u if value in rated_k]

dtotal = 0

for product_id in intersection:
    print(product_id)
    filt = (data_rating_df['user_id'] == user1) & (data_rating_df['product_id'] == product_id)
    # user_i = 
    values1 = data_rating_df[filt]['review_rating'].values[0]
    # values1 = np.array((hitung, 2))
    filt = (data_rating_df['user_id'] == user2) & (data_rating_df['product_id'] == product_id)
    values2 = data_rating_df[filt]['review_rating'].values[0]
    # values2 = np.array((hitung2, 5))
    d = math.sqrt((values1-values2) ** 2)
    dtotal += d
    
    
distance = (1/len(intersection)) * dtotal

