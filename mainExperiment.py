#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:24:16 2021

@author: dhiaaziz
"""

import pandas as pd
import numpy as np
from numpy import int64
import statistics
import os
import time
start_time = time.time()

path = os.getcwd()
folder_dir = path + '/outputs/'
train_file = folder_dir + 'train_set.csv'
print(train_file)
data_rating_df = pd.read_csv(train_file)

user_ids_all = set(data_rating_df.user_id)
product_ids = set(data_rating_df.product_id)


table = pd.pivot_table(data_rating_df, index='user_id', columns='product_id', values='review_rating')
# tableFiltered = table.copy(deep=True)
tableFiltered = table.fillna(0)
tableFiltered = tableFiltered.astype(int)
tableFilteredTrans = tableFiltered.T

#output for anylizing matrices
filedir = path+'/outputs/data.csv' # user-row
tableFiltered.to_csv(filedir, index=True, sep=';')
filedir = path+'/outputs/data_transposed.csv' # product-row
tableFilteredTrans.to_csv(filedir, index=True, sep=';')



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

correlation_data_collection = []
# correlation_df = pd.DataFrame(columns=['user_id', 'user_neighbor_id', 'correlation_coefficient', 'rank'])
# df_temp = pd.DataFrame(data=[[1,1,1,1]], columns=['user_id', 'user_neighbor_id', 'correlation_coefficient', 'rank'])
# correlation_data = correlation_data.append(df_temp, ignore_index=True)
# correlation_data.append([])

for i in tableFiltered.index:
    
    # for experiment purpose, get only 5 users
    # if (i >= 1000):
    #     break
    
    print(f'Calculate Correlation for User {i}')
    series1 = tableFiltered.loc[i] 
    #iterasi sejumlah user id
    for j in tableFiltered.index:
        if j == i:
            continue
        series2 = tableFilteredTrans.loc[:,j]
        corr = series1.corr(series2, method='pearson')
        # correlation_data.append(corr)
        corr_data = {'user_id': i, 'user_neighbor_id': j, 'correlation_coefficient': corr}
        # correlation_df = correlation_df.append(corr_data, ignore_index=True)
        correlation_data_collection.append(corr_data)
        
correlation_df = pd.DataFrame(correlation_data_collection)


    
correlation_df['rank'] = correlation_df.groupby('user_id')['correlation_coefficient'].rank(ascending=False)

# correlationn = tableFiltered.corr(method='pearson')



# ==============================
# Step 2 - Populasi Rating Neighbor
#   desc:
#       koleksi rating yang diberikan oleh 30 tetangga terdekat, dengan cara filter correlation_data kolom rank < 30

#   output:
#       neighbor_rating (pd.DataFrame) - user_id|user_neighbor_id|rank|product_id|neighbor_rating

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


print('Neighbor\'s Rating')
N = 5#neighbor count
correlation_df_filtered = correlation_df[correlation_df['rank'] <= N]

correlation_df_filtered = correlation_df_filtered.reset_index()
correlation_df_filtered = correlation_df_filtered.drop(columns = ['index'])
correlation_df_filtered['user_id'] = correlation_df_filtered['user_id'].astype(int)
correlation_df_filtered['user_neighbor_id'] = correlation_df_filtered['user_neighbor_id'].astype(int)
correlation_df_filtered['rank'] = correlation_df_filtered['rank'].astype(int)

user_ids = sorted(set(correlation_df_filtered.user_id))

neighbor_rating_collection = []
# neighbor_rating_df = pd.DataFrame(columns=['user_id', 'user_neighbor_id', 'rank', 'product_id', 'neighbor_rating'])
for user_id in user_ids:
    neighbor_per_user_df = correlation_df_filtered[correlation_df_filtered['user_id'] == user_id]
    
    for neighbor_id in neighbor_per_user_df['user_neighbor_id']:
        print(f'Collecting Rating from User {user_id} and Neighbor {neighbor_id}')
        rankValue = neighbor_per_user_df.loc[(neighbor_per_user_df.user_id == user_id) & (neighbor_per_user_df.user_neighbor_id == neighbor_id)]['rank']
        rankValue = rankValue.item()
        
        for product_id in product_ids:
            # print(f'product {product_id}')
            neighbor_rating = tableFilteredTrans[neighbor_id].loc[product_id] # tableFilteredTrans[column].loc[index]
            neighbor_data = {'user_id': user_id, 'user_neighbor_id': neighbor_id, 'rank': rankValue , 'product_id': product_id, 'neighbor_rating': neighbor_rating}
            neighbor_rating_collection.append(neighbor_data)
            # neighbor_rating_df = neighbor_rating_df.append(neighbor_data, ignore_index=True)
neighbor_rating_df = pd.DataFrame(neighbor_rating_collection)
        
# print(tableFilteredTrans[21, 67])
# oke = tableFilteredTrans[21]
# oke2 = tableFilteredTrans[21].loc[89]


# ==============================
# Step 3 - Hitung Rekomendasi
#   desc:
#       hitung agregat rating dari semua neighbor

#   output:
#       predicted_rating (pd.DataFrame) - user_id|product_id|predicted_score

#   sample:
#       user_id|product_id|predicted_score
#       user750|product_1|0.333
#       user750|product_2|3
#       user750|product_3|4
#       user67|product_1|0.
#       user67|product_2|3.
#       user67|product_3|4.
# ==============================
test_file = folder_dir + 'test_set.csv'
test_set_df = pd.read_csv(test_file) 

cobacoba = []
list_mean = []
start_time = time.time()
for i in test_set_df.index:
    print(i)
    user_i_predict = test_set_df.iloc[i].user_id
    product_i_predict = test_set_df.iloc[i].product_id
    # user_predict_df = neighbor_rating_df[(neighbor_rating_df['user_id'] == user_i_predict) & (neighbor_rating_df['product_id'] ==  product_i_predict) ]
    # user_predict_df = neighbor_rating_df.query('user_id == @user_i_predict & product_id == @product_i_predict' )
    user_predict_df = neighbor_rating_df.loc[(neighbor_rating_df['user_id'] == user_i_predict) & (neighbor_rating_df['product_id'] ==  product_i_predict)]
    predicted_rating = user_predict_df['neighbor_rating'].mean()
    list_mean.append(predicted_rating)
    # idx = np.where((neighbor_rating_df['user_id'] == user_i_predict) & (neighbor_rating_df['product_id'] == product_i_predict))
    # user_predict_df = neighbor_rating_df.loc[idx]
    # user_predict_df = neighbor_rating_df[neighbor_rating_df.eval("user_id == @user_i_predict & product_id == @product_i_predict")]
test_set_df['predicted_rating'] = list_mean
print("--- %s seconds ---" % (time.time() - start_time))
    
predicted_ratings = []
for user_id in user_ids: 
    print(f'Predict items rating for user: {user_id} ')
    temp_user_df = neighbor_rating_df[neighbor_rating_df['user_id'] == user_id]
    for product_id in sorted(product_ids, reverse=True):
        # print(product_id)
        temp_product_df = temp_user_df[temp_user_df['product_id'] == product_id]
        rating_series = temp_product_df['neighbor_rating']
        predicted_rating = rating_series.mean()
        predicted_data = {'user_id': user_id, 'product_id': product_id, 'predicted_score': predicted_rating}
        predicted_ratings.append(predicted_data)
        

predicted_rating_df = pd.DataFrame(predicted_ratings)


#for development testing purpose => verify data with the results from the excel process
verify_predicted_rating_df = predicted_rating_df[predicted_rating_df['user_id'] == 750]
#verified


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
print("Calculating MAE")
list_mae = []
for user_id in user_ids:
    temp_user_error_df = predicted_rating_df[predicted_rating_df['user_id'] == user_id].sort_values(by=['product_id'])
    temp_actual_rating_series = tableFiltered.loc[user_id].sort_index()

    temp_predicted_rating_series = temp_user_error_df.loc[:,['product_id','predicted_score']].set_index('product_id')
    temp_predicted_rating_series = temp_predicted_rating_series.loc[:, 'predicted_score']

    #pengurangan series / matrix harus memiliki index yang sama
    temp_errors = temp_predicted_rating_series - temp_actual_rating_series
    temp_absolute_errors = temp_errors.abs()
    temp_mae = temp_absolute_errors.mean()
    list_mae.append(temp_mae)
    
mean_absolute_error = statistics.mean(list_mae)
   
#for development testing purpose => verify data with the results from the excel process
# verify_user_error_df = predicted_rating_df[predicted_rating_df['user_id'] == 750].sort_values(by=['product_id'])
# verify_actual_rating_series = tableFiltered.loc[750].sort_index()

# verify_predicted_rating_series = verify_user_error_df.loc[:,['product_id','predicted_score']].set_index('product_id')
# verify_predicted_rating_series = verify_predicted_rating_series.loc[:, 'predicted_score']

#pengurangan series / matrix harus memiliki index yang sama
# verify_errors = verify_predicted_rating_series - verify_actual_rating_series
# verify_absolute_errors = verify_errors.abs()
# verify_mae = verify_absolute_errors.mean()


print("--- %s seconds ---" % (time.time() - start_time))