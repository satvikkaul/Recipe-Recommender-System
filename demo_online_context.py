import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our engines
from backend.inference import RecommenderEngine, ImageEngine

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
IMAGE_MODEL_PATH = "models/saved/image_model_pytorch.pth"
RECOMMENDER_PATH = "models/saved/recommender_model_pytorch.pth"
RECIPES_CSV = "data/food.com-interaction/RAW_recipes.csv"
DATA_DIR_FOOD101 = "data/food-101/images"

def main():
    print("=== NutriSnap Online Context Demo ===")
    
    # 1. Load Engines
    # -------------------------------------------------------
    # Setup Class Names (Same logic as main.py)
    if os.path.exists(DATA_DIR_FOOD101):
        class_names = sorted([d for d in os.listdir(DATA_DIR_FOOD101) if os.path.isdir(os.path.join(DATA_DIR_FOOD101, d))])
    else:
        class_names = ['burger', 'pasta', 'pizza', 'salad', 'sushi']
        
    print(f"\n1. Initializing Engines...")
    if not os.path.exists(IMAGE_MODEL_PATH) or not os.path.exists(RECOMMENDER_PATH):
        print("Error: Models not found. Please train them first.")
        return

    img_engine = ImageEngine(IMAGE_MODEL_PATH, class_names)
    rec_engine = RecommenderEngine(RECOMMENDER_PATH, RECIPES_CSV)
    
    # 2. Simulate User Upload
    # -------------------------------------------------------
    # Let's pick a random image from food-101 or use a placeholder
    print(f"\n2. Simulating User Upload...")
    
    # Try to find a real image
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        # Create a dummy image or look in dataset
        if os.path.exists(DATA_DIR_FOOD101):
            # Pizza folder?
            pizza_dir = os.path.join(DATA_DIR_FOOD101, "pizza")
            if os.path.exists(pizza_dir):
                files = os.listdir(pizza_dir)
                if files:
                    test_image_path = os.path.join(pizza_dir, files[0])
    
    if not os.path.exists(test_image_path):
        print("No test image found. Using random noise for simulation.")
        # Create random image
        params = np.random.randint(0, 255, (260, 260, 3), dtype=np.uint8)
        img = Image.fromarray(params)
    else:
        img = Image.open(test_image_path)
        
    print(f"   Input Image: {test_image_path} (Size: {img.size})")
    
    # Convert to bytes for engine
    import io
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_bytes = buf.getvalue()
    
    # 3. Predict Class
    # -------------------------------------------------------
    print(f"\n3. Predicting Food Class...")
    pred_result = img_engine.predict(img_bytes)
    food_name = pred_result['food_name']
    confidence = pred_result['confidence']
    print(f"   Detected: **{food_name.upper()}** (Conf: {confidence:.2f})")
    print(f"   Nutrition Est: {pred_result['nutrition']}")
    
    # 4. Contextual Recommendation
    # -------------------------------------------------------
    print(f"\n4. Generating Context-Aware Recommendations...")
    print(f"   (Searching for '{food_name}' in recipe database to use as Anchor...)")
    
    # Find a recipe ID that matches this name query
    # Simple substring search
    mask = rec_engine.recipes_df['name'].str.contains(food_name, case=False, na=False)
    matches = rec_engine.recipes_df[mask]
    
    if matches.empty:
        print(f"   Warning: No specific recipe found for '{food_name}'. Using random anchor.")
        anchor_idx = 0 
    else:
        # Pick the most popular one? Or just the first one.
        # Let's pick one with a high ID count if possible, or just first.
        anchor_row = matches.iloc[0]
        anchor_name = anchor_row['name']
        anchor_id = str(anchor_row['recipe_id'])
        print(f"   Anchor Recipe Found: '{anchor_name}' (ID: {anchor_id})")
        
        # Get internal index
        if anchor_id in rec_engine.recipe_to_idx:
            anchor_idx = rec_engine.recipe_to_idx[anchor_id]
        else:
             print("   Anchor ID not in model vocabulary. Fallback.")
             anchor_idx = 0
             
    # 5. Embedding Space Search
    # -------------------------------------------------------
    # Get Embedding [1, Dim]
    anchor_emb = rec_engine.item_embeddings[anchor_idx].unsqueeze(0) # [1, Dim]
    
    # Dot Product with all other items
    # Note: In Two-Tower, high dot product means "Likely to be interacted with by same user".
    # It effectively finds "Complementary" or "Similar" items depending on data.
    scores = torch.matmul(anchor_emb, rec_engine.item_embeddings.T).squeeze(0)
    
    # Top K
    top_k = 5
    top_indices = torch.topk(scores, k=top_k+1).indices.cpu().numpy()
    
    print(f"\n   --- Recommendations based on eating {food_name.upper()} ---")
    for idx in top_indices:
        if idx == anchor_idx: continue # Skip itself
        
        rid = rec_engine.idx_to_recipe[idx]
        row = rec_engine.recipes_df[rec_engine.recipes_df['recipe_id'] == rid].iloc[0]
        print(f"   [Rec] {row['name']} (Cal: {row['calories']})")

if __name__ == "__main__":
    main()
