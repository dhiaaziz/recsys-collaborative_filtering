#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 02:33:19 2021

@author: dhiaaziz
"""


import pandas as pd
import os 



in_df = pd.read_json("./datasets/top10cat_balanced.json", lines=True)
in_df = in_df[['review_id','product_id','user_id','review_rating','review_packaging','review_rebuy','review_valueformoney']]
user_ids_before = set(in_df['user_id'])
product_ids_before = set(in_df['product_id'])

in_df = in_df.groupby('user_id').filter(lambda data: len(data) >= 5)
user_ids = set(in_df['user_id'])
product_ids = set(in_df['product_id'])

multi_rating_df = in_df.loc[:][['user_id', 'product_id', 'review_packaging','review_rebuy','review_valueformoney']]
multi_rating_df = multi_rating_df.sort_values(by=['user_id', 'product_id'], ascending=[True, True])

train_df = multi_rating_df.sample(frac=0.8,random_state=200)
train_df = train_df.sort_values(by="user_id")
test_df = multi_rating_df.drop(train_df.index)
user_ids_train = set(train_df['user_id'])
product_ids_train = set(train_df['product_id'])
user_ids_test = set(test_df['user_id'])
product_ids_test = set(test_df['product_id'])

product_ids_onlytest_pre =  product_ids_test - product_ids_train#find product ids not in that exist in test but not in train

path = os.getcwd()
folder_dir = path + '/outputs'
file_dir = folder_dir + '/train_set_multi.csv'
# print(file_dir)

train_df.to_csv(file_dir, index=False, sep=",")

file_dir = folder_dir + '/test_set_multi.csv'
test_df.to_csv(file_dir, index=False, sep=",")

file_dir = folder_dir + '/product_ids_only_in_test.csv'
# product_ids_onlytest.to