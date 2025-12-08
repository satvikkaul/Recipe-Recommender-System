#!/usr/bin/env python
# coding: utf-8

# # Train Recommender System (Two-Tower)
# 
# This notebook trains the TFRS model on the mock data and saves the BruteForce index for serving.

# In[ ]:


import os
import sys
import tensorflow as tf

# Add root to path so we can import models
sys.path.append(os.path.abspath('..'))

from models.recommender import build_model, save_model_index


# In[ ]:


users_csv = '../data/mock/users.csv'
recipes_csv = '../data/mock/recipes.csv'
interactions_csv = '../data/mock/interactions.csv'

model, train_ds = build_model(users_csv, recipes_csv, interactions_csv)

print("Training model...")
model.fit(train_ds, epochs=5)


# In[ ]:


export_path = '../models/saved/recommender_index'
save_model_index(model, recipes_csv, export_path)


# In[ ]:


# Test the saved index
loaded = tf.saved_model.load(export_path)

# Predict for a dummy user
scores, titles = loaded(["user_1"])

print(f"Recommendations for user_1: {titles[0].numpy()}")

