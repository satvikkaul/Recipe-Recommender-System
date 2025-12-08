import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd

# Define paths
# Note: we need to import our model definitions if we were loading weights into the class, 
# but for SavedModel, we load the graph directly.

class RecommenderEngine:
    def __init__(self, model_path, recipes_csv):
        print(f"Loading Recommender from {model_path}...")
        self.model = tf.saved_model.load(model_path)
        self.recipes_df = pd.read_csv(recipes_csv)
        
        # Handle renaming
        if "id" in self.recipes_df.columns and "recipe_id" not in self.recipes_df.columns:
            self.recipes_df = self.recipes_df.rename(columns={"id": "recipe_id"})
            
        self.recipes_df['recipe_id'] = self.recipes_df['recipe_id'].astype(str)
        
        # Parse nutrition to extract calories (first element of list string)
        # Format: "[calories, fat, sugar, sodium, protein, sat_fat, carbs]"
        def get_calories(val):
            try:
                # Remove brackets and split
                val = val.strip("[]")
                parts = val.split(",")
                return float(parts[0])
            except:
                return 0.0
                
        if 'nutrition' in self.recipes_df.columns:
            self.recipes_df['calories'] = self.recipes_df['nutrition'].apply(get_calories)
            
        print("Recommender loaded.")

    def recommend(self, user_id, current_calories, daily_goal=2000, top_k=10):
        """
        1. Get candidates from TFRS model.
        2. Filter/Re-rank based on remaining calorie budget.
        """
        remaining_budget = daily_goal - current_calories
        
        # Get candidates (returns scores, ids)
        # Input to model depends on signature. Our saved index expects user_id string tensor.
        _, candidate_ids = self.model(tf.constant([str(user_id)]))
        
        # Convert tensors to numpy strings
        candidate_ids = candidate_ids[0].numpy().astype(str)
        
        recommendations = []
        
        for rid in candidate_ids:
            # Look up recipe details
            row = self.recipes_df[self.recipes_df['recipe_id'] == rid]
            if row.empty:
                continue
            
            recipe_info = row.iloc[0].to_dict()
            
            # Simple Logic: Check if it fits within the budget (plus a small buffer)
            # Or rank by how well it fits.
            # Here we just mark it as "fits_budget"
            recipe_info['fits_budget'] = recipe_info['calories'] <= (remaining_budget + 100)
            
            recommendations.append(recipe_info)
            
        # Re-ranking logic:
        # Prioritize recipes that fit the budget.
        # Within that, maybe prioritize healthy macros (e.g. high protein) - optional extension.
        
        # Sort by: Fits Budget (True first), then maybe by protein (descending) or just keep similarity order
        # Let's keep similarity order but put budget-fitting ones on top.
        
        recommendations.sort(key=lambda x: x['fits_budget'], reverse=True)
        
        return recommendations[:top_k]

class ImageEngine:
    def __init__(self, model_path, class_names):
        print(f"Loading Image Model from {model_path}...")
        # Load as Keras model (handles both .keras and SavedModel formats)
        if model_path.endswith('.keras'):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = tf.saved_model.load(model_path)
        self.class_names = class_names
        # Mock nutritional info for classes
        self.nutrition_db = {
            "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10},
            "burger": {"calories": 354, "protein": 17, "carbs": 31, "fat": 17},
            "salad": {"calories": 150, "protein": 5, "carbs": 10, "fat": 10},
            "sushi": {"calories": 200, "protein": 9, "carbs": 28, "fat": 5},
            "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1},
            # Default fallback
            "unknown": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        }
        print("Image Model loaded.")

    def predict(self, image_bytes):
        # Preprocess image
        # Resize to 224x224
        img = tf.io.decode_image(image_bytes, channels=3)
        img = tf.image.resize(img, [224, 224])
        # Add batch dimension
        img = tf.expand_dims(img, 0)
        
        # Predict
        # Note: The SavedModel might have a specific signature.
        # For a standard Sequential model saved via tf.saved_model.save, 
        # it's usually just calling the object.
        predictions = self.model(img)
        
        score = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(score)
        predicted_class = self.class_names[class_idx]
        
        nutrition = self.nutrition_db.get(predicted_class, self.nutrition_db["unknown"])
        
        return {
            "food_name": predicted_class,
            "confidence": float(np.max(score)),
            "nutrition": nutrition
        }
