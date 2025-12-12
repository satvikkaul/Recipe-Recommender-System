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
    def __init__(self, model_path, class_names, recipes_csv=None):
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
        
        # Build nutrition database from recipes CSV
        self.nutrition_db = self._build_nutrition_db(recipes_csv)
        print("Image Model loaded.")

    def _build_nutrition_db(self, recipes_csv):
        """
        Build nutrition lookup from recipes CSV.
        Maps food names to average nutrition values from actual recipes.
        Nutrition format: [calories, fat, sugar, sodium, protein, saturated_fat, carbs]
        """
        nutrition_db = {}
        
        if recipes_csv and os.path.exists(recipes_csv):
            try:
                print(f"Loading nutrition data from {recipes_csv}...")
                recipes = pd.read_csv(recipes_csv)
                
                # Parse nutrition column
                recipes['nutrition'] = recipes['nutrition'].apply(parse_list_column)
                
                # Group by food name and compute average nutrition
                for food_name in self.class_names:
                    # Find recipes that match this food category
                    # Match by looking for food name in recipe name (approximate matching)
                    matching_recipes = recipes[
                        recipes['name'].str.lower().str.contains(food_name.replace('_', ' '), na=False, case=False)
                    ]
                    
                    if len(matching_recipes) > 0:
                        # Get nutrition values (format: [Cal, Fat, Sugar, Sodium, Protein, SatFat, Carbs])
                        nutrition_lists = []
                        for nut in matching_recipes['nutrition']:
                            if isinstance(nut, list) and len(nut) >= 7:
                                nutrition_lists.append(nut)
                        
                        if nutrition_lists:
                            # Average the nutrition values
                            avg_nutrition = np.mean(nutrition_lists, axis=0)
                            nutrition_db[food_name] = {
                                "calories": int(avg_nutrition[0]),
                                "fat": round(avg_nutrition[1], 1),
                                "sugar": round(avg_nutrition[2], 1),
                                "sodium": int(avg_nutrition[3]),
                                "protein": round(avg_nutrition[4], 1),
                                "saturated_fat": round(avg_nutrition[5], 1),
                                "carbs": round(avg_nutrition[6], 1),
                            }
                    
                print(f"Built nutrition DB for {len(nutrition_db)} food categories from recipes")
            except Exception as e:
                print(f"Warning: Could not load nutrition from CSV: {e}")
        
        # Return nutrition_db, or empty if loading failed
        return nutrition_db

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_idx.item()]
        
        # Get nutrition, with fallback to zeros if not found
        nutrition = self.nutrition_db.get(
            predicted_class, 
            {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        )
        
        return {
            "food_name": predicted_class,
            "confidence": float(confidence.item()),
            "nutrition": nutrition
        }

class RecommenderEngine:
    def __init__(self, model_path, recipes_csv, interactions_csv=None, max_history=15):
        print(f"Loading Recommender from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history = max_history

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

        # 5. Load User Interaction History (optional but recommended)
        self.user_history = {}
        if interactions_csv and os.path.exists(interactions_csv):
            print("Loading user interaction history...")
            self._load_user_history(interactions_csv)
            print(f"Loaded history for {len(self.user_history)} users.")
        else:
            print("No interaction history provided. User embeddings will be ID-only.")

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

    def _load_user_history(self, interactions_csv):
        """Load user interaction history from interactions CSV"""
        interactions_df = pd.read_csv(interactions_csv)
        interactions_df['user_id'] = interactions_df['user_id'].astype(str)
        interactions_df['recipe_id'] = interactions_df['recipe_id'].astype(str)

        # Filter to known recipes only
        known_recipes = set(self.recipe_to_idx.keys())
        interactions_df = interactions_df[interactions_df['recipe_id'].isin(known_recipes)]

        # Sort by date if available
        if 'date' in interactions_df.columns:
            interactions_df = interactions_df.sort_values('date')

        # Group by user and collect most recent interactions
        for user_id, group in interactions_df.groupby('user_id'):
            if user_id in self.user_to_idx:
                # Get most recent max_history recipes
                recipe_ids = group['recipe_id'].tail(self.max_history).tolist()
                # Store as internal indices
                recipe_indices = [self.recipe_to_idx[rid] for rid in recipe_ids if rid in self.recipe_to_idx]
                self.user_history[user_id] = recipe_indices

    def _get_user_history_embeddings(self, user_id):
        """Get history embeddings for a user"""
        if user_id not in self.user_history or len(self.user_history[user_id]) == 0:
            return None, None

        # Get recipe indices from history
        history_indices = self.user_history[user_id]

        # Get embeddings from precomputed item embeddings
        history_embs = self.item_embeddings[history_indices]  # [H, D]

        # Add batch dimension and create mask
        history_embs = history_embs.unsqueeze(0)  # [1, H, D]
        history_mask = torch.ones(1, len(history_indices), dtype=torch.bool, device=self.device)

        return history_embs, history_mask

    def recommend(self, user_id, current_calories, daily_goal=2000, top_k=10):
        if user_id not in self.user_to_idx:
            print(f"Unknown user {user_id}, using popularity-based recommendations")
            # Fallback to content-based recommendations for unknown users
            return self._recommend_for_new_user(current_calories, daily_goal, top_k)

        user_idx = self.user_to_idx[user_id]
        user_tensor = torch.tensor([user_idx], device=self.device)

        # Get user history embeddings
        history_embs, history_mask = self._get_user_history_embeddings(user_id)

        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_tensor, history_embs, history_mask)

        # Get base scores from model
        scores = torch.matmul(user_emb, self.item_embeddings.T).squeeze(0)
        
        # Calculate remaining calorie budget
        remaining_budget = daily_goal - current_calories
        
        # GOAL-AWARE RE-RANKING: Boost scores for items that fit budget
        # This implements the proposal requirement: "recipes that fit remaining daily budget"
        for idx in range(len(scores)):
            recipe_id = self.idx_to_recipe[idx]
            row = self.recipes_df[self.recipes_df['recipe_id'] == recipe_id]
            
            if not row.empty:
                calories = row.iloc[0]['calories']
                
                # Calculate budget fitness score (0 to 1)
                # Perfect score when calories match budget exactly
                # Decreases as difference increases
                if remaining_budget > 0:
                    budget_fit = max(0, 1 - abs(calories - remaining_budget) / max(remaining_budget, 500))
                    # Boost the score by weighted budget fitness (0.3 weight for goal-awareness)
                    scores[idx] = scores[idx] + 0.3 * budget_fit
        
        # Get top candidates after re-ranking
        top_indices = torch.topk(scores, k=top_k*5).indices.cpu().numpy()

        recommendations = []

        # Map internal indices back to data
        for idx in top_indices:
            recipe_id = self.idx_to_recipe[idx]

            # Lookup metadata
            row = self.recipes_df[self.recipes_df['recipe_id'] == recipe_id]
            if row.empty: continue

            info = row.iloc[0].to_dict()
            info['fits_budget'] = info['calories'] <= (remaining_budget + 100)
            info['score'] = float(scores[idx].cpu())  # Add score for debugging

            recommendations.append(info)
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _recommend_for_new_user(self, current_calories, daily_goal, top_k=10):
        """
        Recommend recipes for new/unknown users based on:
        1. Budget fit (remaining calories)
        2. Popularity (random selection from healthy options)
        3. Nutritional balance
        """
        remaining_budget = daily_goal - current_calories
        
        # Filter recipes that fit budget (with 100 cal tolerance)
        budget_recipes = self.recipes_df[
            self.recipes_df['calories'] <= (remaining_budget + 100)
        ].copy()
        
        # If no recipes fit, get lowest calorie recipes
        if budget_recipes.empty:
            budget_recipes = self.recipes_df.nsmallest(top_k * 2, 'calories').copy()
        
        # Score by nutritional balance and calorie fit
        budget_recipes['score'] = 1.0 / (1.0 + abs(budget_recipes['calories'] - remaining_budget/2))
        
        # Sort by score and take top_k
        budget_recipes = budget_recipes.sort_values('score', ascending=False)
        
        recommendations = []
        for _, row in budget_recipes.head(top_k * 2).iterrows():
            info = row.to_dict()
            info['fits_budget'] = info['calories'] <= (remaining_budget + 100)
            info['score'] = float(info['score'])
            recommendations.append(info)
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
