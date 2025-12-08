import os
import pprint
import tempfile
from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Define the Recommender Model
class TwoTowerModel(tfrs.Model):
    def __init__(self, user_model, item_model, task):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the item features and pass them into the item model,
        # getting embeddings which will be used to compute the prediction.
        recipe_embeddings = self.item_model(features["recipe_id"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, recipe_embeddings)

class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32)
        ])

    def call(self, inputs):
        return self.user_embedding(inputs)

class RecipeModel(tf.keras.Model):
    def __init__(self, unique_recipe_ids):
        super().__init__()
        self.recipe_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_recipe_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_recipe_ids) + 1, 32)
        ])
        # In a more advanced version, we would concatenate this with 
        # normalized nutritional features here. For this baseline Two-Tower,
        # we focus on the ID embedding.
        
    def call(self, inputs):
        return self.recipe_embedding(inputs)

def build_model(users_csv, recipes_csv, interactions_csv):
    # Load data
    users = pd.read_csv(users_csv)
    recipes = pd.read_csv(recipes_csv)
    interactions = pd.read_csv(interactions_csv)

    # Preprocessing for Real Data compatibility
    if "id" in recipes.columns and "recipe_id" not in recipes.columns:
        recipes = recipes.rename(columns={"id": "recipe_id"})
        
    # Ensure ID columns are treated as strings
    recipes["recipe_id"] = recipes["recipe_id"].astype(str)
    interactions["recipe_id"] = interactions["recipe_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    
    # Get unique vocabularies
    unique_user_ids = users["user_id"].unique().astype(str)
    unique_recipe_ids = recipes["recipe_id"].unique().astype(str)

    # Create datasets
    # Only keep the columns we need to avoid issues with other columns (like mixed types in reviews)
    interactions_subset = interactions[["user_id", "recipe_id"]]
    interactions_ds = tf.data.Dataset.from_tensor_slices(dict(interactions_subset))
    interactions_ds = interactions_ds.map(lambda x: {
        "user_id": tf.cast(x["user_id"], tf.string),
        "recipe_id": tf.cast(x["recipe_id"], tf.string),
    })
    
    
    # Same precaution for recipes, we only need recipe_id for the dataset map below
    recipes_subset = recipes[["recipe_id"]]
    recipes_ds = tf.data.Dataset.from_tensor_slices(dict(recipes_subset))
    recipes_ds = recipes_ds.map(lambda x: tf.cast(x["recipe_id"], tf.string))
    
    # Shuffle and batch
    # Shuffle and batch
    # Optimized for speed: larger batch size (8192) and smaller shuffle buffer (10000)
    cached_train = interactions_ds.shuffle(10_000).batch(8192).cache()
    cached_recipes = recipes_ds.batch(1024).cache()

    # Define Towers
    user_model = UserModel(unique_user_ids)
    item_model = RecipeModel(unique_recipe_ids)
    
    # Define Task (Retrieval)
    metrics = tfrs.metrics.FactorizedTopK(
        candidates=cached_recipes.map(item_model)
    )
    
    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )

    # Instantiate Model
    model = TwoTowerModel(user_model, item_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    
    return model, cached_train

def save_model_index(model, recipes_csv, export_path):
    """
    Saves a BruteForce layer that can take a user_id and return top K recommendations.
    This is what we will serve in the API.
    """
    recipes = pd.read_csv(recipes_csv)
    
    # Preprocessing for Real Data compatibility
    if "id" in recipes.columns and "recipe_id" not in recipes.columns:
        recipes = recipes.rename(columns={"id": "recipe_id"})
        
    unique_recipe_ids = recipes["recipe_id"].unique().astype(str)
    
    # We need to recreate the item model structure to pass it to the BruteForce layer
    # or just use the trained model's item_model if accessible.
    
    # Create the index
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    
    # Create dataset of candidates for the index
    # We need (identifier, embedding) pairs or just embeddings if we track identifiers separately
    # But TFRS BruteForce.index_from_dataset usually takes a dataset of candidate embeddings
    # AND optionally candidate identifiers.
    
    recipes_ds = tf.data.Dataset.from_tensor_slices(unique_recipe_ids)
    
    # We zip (recipe_id, recipe_embedding)
    # Note: model.item_model takes recipe_id -> embedding
    
    index.index_from_dataset(
        tf.data.Dataset.zip((recipes_ds.batch(100), recipes_ds.batch(100).map(model.item_model)))
    )
    
    # Run once to build the layer and trace the graph
    _ = index(tf.constant(["user_1"]))

    # Save
    tf.saved_model.save(index, export_path)
    print(f"Model index saved to {export_path}")

if __name__ == "__main__":
    pass
