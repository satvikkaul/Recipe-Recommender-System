import torch
import numpy as np
import pandas as pd
from PIL import Image
import io
import sys
import os
import ast
from torchvision import transforms

# Ensure we can import from models
sys.path.append(os.getcwd())
from models.image_classifier import build_model as build_image_model
from models.recommender import TwoTowerModel

# Re-use parsing logic (Duplication for standalone robustness)
def parse_list_column(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

class ImageEngine:
    def __init__(self, model_path, class_names):
        print(f"Loading Image Model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
            
        self.model = build_image_model(len(self.class_names), device=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # EfficientNetB2 expects 260x260
        self.transforms = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
        
        self.nutrition_db = {
            "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10},
            "burger": {"calories": 354, "protein": 17, "carbs": 31, "fat": 17},
            "salad": {"calories": 150, "protein": 5, "carbs": 10, "fat": 10},
            "sushi": {"calories": 200, "protein": 9, "carbs": 28, "fat": 5},
            "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1},
        }
        print("Image Model loaded.")

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_idx.item()]
        
        nutrition = self.nutrition_db.get(predicted_class, {"calories": 0, "protein": 0, "carbs": 0, "fat": 0})
        
        return {
            "food_name": predicted_class,
            "confidence": float(confidence.item()),
            "nutrition": nutrition
        }

class RecommenderEngine:
    def __init__(self, model_path, recipes_csv):
        print(f"Loading Recommender from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Data
        self.recipes_df = pd.read_csv(recipes_csv)
        if "id" in self.recipes_df.columns and "recipe_id" not in self.recipes_df.columns:
            self.recipes_df = self.recipes_df.rename(columns={"id": "recipe_id"})
        self.recipes_df['recipe_id'] = self.recipes_df['recipe_id'].astype(str)
        
        # Parse content columns
        self.recipes_df['ingredients'] = self.recipes_df['ingredients'].apply(parse_list_column)
        self.recipes_df['nutrition'] = self.recipes_df['nutrition'].apply(parse_list_column)
        
        # Simple calorie parse (first element) for filtering
        def get_calories(row):
            try:
                if len(row) > 0: return float(row[0])
            except: pass
            return 0.0
        self.recipes_df['calories'] = self.recipes_df['nutrition'].apply(get_calories)

        # 2. Load Checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        self.user_to_idx = checkpoint['user_to_idx']
        self.recipe_to_idx = checkpoint['recipe_to_idx'] # Map ID -> Internal Index
        self.idx_to_recipe = {v: k for k, v in self.recipe_to_idx.items()}
        self.vocab = checkpoint['vocab']
        embedding_dim = checkpoint['embedding_dim']
        
        # 3. Build Model
        self.model = TwoTowerModel(
            num_users=len(self.user_to_idx),
            num_recipes=len(self.recipe_to_idx),
            vocab_size=len(self.vocab),
            embedding_dim=embedding_dim
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 4. Precompute Item Embeddings
        # This is strictly more complex now because we need to feed content features
        # We process all recipes into tensors once
        print("Precomputing recipe embeddings (content-aware)...")
        self.item_embeddings = self._precompute_all_items()
        print("Recommender loaded.")

    def _precompute_all_items(self):
        # We need to run the RecipeTower for ALL recipes in order (0..N)
        # 1. Prepare Tensors
        all_indices = []
        all_ingredients = []
        all_nutrition = []
        
        unk_idx = 0
        scale = np.array([1000, 100, 100, 1000, 100, 100, 100], dtype=np.float32)
        
        # Iterate in Order of Internal Index (0, 1, 2...)
        # This assumes recipe_to_idx works for every row in df or we map appropriately
        # Safer to create a map: internal_idx -> row
        
        # Create a fast lookup map for dataframe rows
        # Assumes recipe_id match
        df_map = self.recipes_df.set_index('recipe_id')
        
        num_items = len(self.recipe_to_idx)
        
        # Lists for batching
        batch_ids = []
        batch_ing = []
        batch_nut = []
        
        final_embeddings = []
        
        # Process in chunks to avoid OOM
        CHUNK_SIZE = 1024
        
        with torch.no_grad():
            for i in range(num_items):
                rid = self.idx_to_recipe[i]
                if rid in df_map.index:
                    row = df_map.loc[rid]
                    # Handle Duplicate IDs in CSV? loc returns maybe multiple. Take first.
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    
                    ing_list = row['ingredients']
                    nut_list = row['nutrition']
                else:
                    ing_list = []
                    nut_list = []
                
                # Process Ing
                ing_idxs = [self.vocab.get(x, unk_idx) for x in ing_list]
                if not ing_idxs: ing_idxs = [unk_idx]
                
                # Process Nut
                nut = np.array(nut_list, dtype=np.float32)
                if len(nut) == 7:
                    nut = nut / scale
                else:
                    nut = np.zeros(7, dtype=np.float32)
                    
                batch_ids.append(i)
                batch_ing.append(torch.tensor(ing_idxs, dtype=torch.long))
                batch_nut.append(torch.tensor(nut, dtype=torch.float32))
                
                if len(batch_ids) >= CHUNK_SIZE or i == num_items - 1:
                    # Collate Batch
                    b_ids = torch.tensor(batch_ids).to(self.device)
                    b_nut = torch.stack(batch_nut).to(self.device)
                    
                    # Pad ingredients
                    max_len = max([len(x) for x in batch_ing])
                    b_ing = torch.zeros((len(batch_ids), max_len), dtype=torch.long).to(self.device)
                    for j, seq in enumerate(batch_ing):
                        b_ing[j, :len(seq)] = seq.to(self.device)
                        
                    # Forward
                    embs = self.model.get_recipe_embedding(b_ids, b_ing, b_nut)
                    final_embeddings.append(embs.cpu())
                    
                    batch_ids = []
                    batch_ing = []
                    batch_nut = []
        
        return torch.cat(final_embeddings, dim=0).to(self.device) # [N, Dim]

    def recommend(self, user_id, current_calories, daily_goal=2000, top_k=10):
        if user_id not in self.user_to_idx:
            print(f"Unknown user {user_id}")
            return []
            
        user_idx = self.user_to_idx[user_id]
        user_tensor = torch.tensor([user_idx], device=self.device)
        
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_tensor)
            
        scores = torch.matmul(user_emb, self.item_embeddings.T).squeeze(0)
        top_indices = torch.topk(scores, k=top_k*5).indices.cpu().numpy()
        
        recommendations = []
        remaining_budget = daily_goal - current_calories
        
        # Map internal indices back to data
        for idx in top_indices:
            recipe_id = self.idx_to_recipe[idx]
            
            # Lookup metadata
            row = self.recipes_df[self.recipes_df['recipe_id'] == recipe_id]
            if row.empty: continue
            
            info = row.iloc[0].to_dict()
            info['fits_budget'] = info['calories'] <= (remaining_budget + 100)
            
            recommendations.append(info)
            if len(recommendations) >= top_k:
                break
                
        return recommendations
