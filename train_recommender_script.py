import os
# Must set this before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import tensorflow as tf
import pandas as pd
import numpy as np

# Ensure we can import models
sys.path.append(os.getcwd())

from models.recommender import build_model, save_model_index

def main():
    print("TF Version:", tf.__version__)
    
    # Updated paths for real Food.com data
    # Note: We use RAW_interactions for both interactions and user list source
    users_csv = 'data/food.com-interaction/RAW_interactions.csv' 
    recipes_csv = 'data/food.com-interaction/RAW_recipes.csv'
    interactions_csv = 'data/food.com-interaction/RAW_interactions.csv'
    
    if not os.path.exists(users_csv):
        print(f"Error: {users_csv} not found.")
        return

    print("Building and training model...")
    # Wrap in try-except to catch the Keras 3 compatibility issue explicitly
    try:
        model, train_ds = build_model(users_csv, recipes_csv, interactions_csv)
        model.fit(train_ds, epochs=3)
        
        export_path = 'models/saved/recommender_index'
        save_model_index(model, recipes_csv, export_path)
        
        # Test loading
        loaded = tf.saved_model.load(export_path)
        scores, titles = loaded([str("user_1")])
        print(f"Recommendations for user_1: {titles[0].numpy()}")
        
    except Exception as e:
        print("An error occurred during training:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
