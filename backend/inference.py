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
        self.demo_users = set()  # Track demo users added at runtime
        if interactions_csv and os.path.exists(interactions_csv):
            print("Loading user interaction history...")
            self._load_user_history(interactions_csv)
            print(f"Loaded history for {len(self.user_history)} users.")
        else:
            print("No interaction history provided. User embeddings will be ID-only.")

        # Diagnostic: Show sample user IDs
        sample_users = list(self.user_to_idx.keys())[:10]
        print(f"Total users in model: {len(self.user_to_idx)}")
        print(f"Sample user IDs: {sample_users}")

        # Warmup: Initialize CUDA context if using GPU
        if self.device.type == 'cuda':
            print("Warming up GPU (initializing CUDA context)...")
            try:
                with torch.no_grad():
                    # Do a small matrix multiplication to initialize CUDA
                    dummy = torch.randn(10, 10, device=self.device)
                    _ = torch.matmul(dummy, dummy.T)
                print("GPU warmup complete")
            except Exception as e:
                print(f"Warning: GPU warmup failed: {e}")

        print("Recommender loaded.")

    def load_demo_users(self, demo_interactions_csv="data/food.com-interaction/demo_users_interactions.csv"):
        """
        Load demo users into the system at runtime without retraining.
        Demo users are tracked separately since they weren't in the training set.
        They will use history-based recommendations only (no user ID embedding).
        """
        if not os.path.exists(demo_interactions_csv):
            print(f"Warning: Demo users file not found: {demo_interactions_csv}")
            return False

        print(f"Loading demo users from {demo_interactions_csv}...")
        demo_df = pd.read_csv(demo_interactions_csv)
        demo_df['user_id'] = demo_df['user_id'].astype(str)
        demo_df['recipe_id'] = demo_df['recipe_id'].astype(str)

        # Get unique demo user IDs
        demo_user_ids = demo_df['user_id'].unique()

        # Track demo users separately (DON'T add to user_to_idx - they're not in the model)
        self.demo_users = set(demo_user_ids)

        # Load demo user interaction history
        known_recipes = set(self.recipe_to_idx.keys())
        for user_id in demo_user_ids:
            user_interactions = demo_df[demo_df['user_id'] == user_id]

            # Filter to known recipes only
            valid_interactions = user_interactions[
                user_interactions['recipe_id'].isin(known_recipes)
            ]

            if len(valid_interactions) > 0:
                # Sort by date (most recent first) and take last max_history
                if 'date' in valid_interactions.columns:
                    valid_interactions = valid_interactions.sort_values('date')

                # Get recipe indices
                recipe_ids = valid_interactions['recipe_id'].tail(self.max_history).tolist()
                recipe_indices = [self.recipe_to_idx[rid] for rid in recipe_ids if rid in self.recipe_to_idx]

                # Store history
                self.user_history[user_id] = recipe_indices
                print(f"  Loaded {len(recipe_indices)} interactions for {user_id}")

        print(f"[OK] Loaded {len(demo_user_ids)} demo users successfully")
        print(f"Note: Demo users will use history+context recommendations (not in trained model)")
        return True

    def get_demo_users(self):
        """Get list of demo users"""
        demo_users_list = []

        # Check if demo_users attribute exists and has content
        if hasattr(self, 'demo_users') and self.demo_users:
            for user_id in self.demo_users:
                history_count = len(self.user_history.get(user_id, []))
                # Extract cuisine name from user_id (e.g., "demo_user_italian" -> "Italian")
                cuisine = user_id.replace("demo_user_", "").title()
                demo_users_list.append({
                    "user_id": user_id,
                    "interactions": history_count,
                    "cuisine": cuisine
                })

        return demo_users_list

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

    def get_user_info(self):
        """Get diagnostic information about available users"""
        return {
            "total_users": len(self.user_to_idx),
            "sample_user_ids": list(self.user_to_idx.keys())[:20],
            "users_with_history": len(self.user_history)
        }

    def user_exists(self, user_id):
        """Check if a user ID exists in the model or is a demo user"""
        user_id = str(user_id)
        # Check if user is in trained model OR is a demo user
        is_demo = hasattr(self, 'demo_users') and user_id in self.demo_users
        return user_id in self.user_to_idx or is_demo

    def _find_anchor_recipe(self, food_name):
        """
        Find a recipe that matches the uploaded food name.
        Uses substring matching on recipe names.
        Returns the internal recipe index.
        """
        # Clean food name (remove underscores, lowercase)
        search_term = food_name.replace('_', ' ').lower().strip()

        # Search in recipe names
        mask = self.recipes_df['name'].str.lower().str.contains(search_term, na=False, case=False)
        if 'tags' in self.recipes_df.columns:
            # Also search tags column (string) to improve matches
            mask = mask | self.recipes_df['tags'].astype(str).str.lower().str.contains(search_term, na=False, case=False)
        matches = self.recipes_df[mask]

        if matches.empty:
            # Fallback: use a deterministic hash so different foods map to different anchors
            num_items = len(self.item_embeddings)
            hashed_idx = hash(search_term) % max(1, num_items)
            print(f"No recipe match for '{food_name}', using hashed anchor index {hashed_idx}")
            return hashed_idx

        # Return the most popular match (or first one)
        anchor_row = matches.iloc[0]
        anchor_id = str(anchor_row['recipe_id'])

        if anchor_id in self.recipe_to_idx:
            return self.recipe_to_idx[anchor_id]
        else:
            return 0

    def recommend(self, user_id, current_calories, daily_goal=2000, top_k=10):
        # Ensure user_id is a string for consistent lookup
        user_id = str(user_id)
        remaining_budget = daily_goal - current_calories

        is_demo_user = hasattr(self, 'demo_users') and user_id in self.demo_users

        if is_demo_user:
            print(f"Demo user '{user_id}' detected in recommend()")
            if user_id in self.user_history and len(self.user_history[user_id]) > 0:
                history_indices = self.user_history[user_id]
                history_embs = self.item_embeddings[history_indices]
                user_pref_emb = history_embs.mean(dim=0, keepdim=True)  # [1, D]
                with torch.no_grad():
                    scores = torch.matmul(user_pref_emb, self.item_embeddings.T).squeeze(0)
            else:
                print("Demo user has no history, using popularity-based recommendations")
                return self._get_popular_recommendations(remaining_budget, top_k)

        elif user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            user_tensor = torch.tensor([user_idx], device=self.device)

            # Get user history embeddings
            history_embs, history_mask = self._get_user_history_embeddings(user_id)

            with torch.no_grad():
                user_emb = self.model.get_user_embedding(user_tensor, history_embs, history_mask)

            # Get base scores from model
            scores = torch.matmul(user_emb, self.item_embeddings.T).squeeze(0)
        else:
            sample_users = list(self.user_to_idx.keys())[:5]
            print(f"Unknown user '{user_id}' (type: {type(user_id).__name__})")
            print(f"Available sample users: {sample_users}")
            print(f"Total users in model: {len(self.user_to_idx)}")
            print("Using popularity-based recommendations as fallback")
            # Fallback to content-based recommendations for unknown users
            return self._get_popular_recommendations(remaining_budget, top_k)
        
        # Calculate remaining calorie budget
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

            # Clean NaN values for JSON serialization
            info = {k: (None if pd.isna(v) else v) for k, v in info.items()}

            # Safely check calorie budget (handle None and convert to scalar)
            cal_value = info.get('calories')
            info['fits_budget'] = bool(cal_value is not None and float(cal_value) <= (remaining_budget + 100))
            info['score'] = float(scores[idx].cpu())  # Add score for debugging

            recommendations.append(info)
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _get_popular_recommendations(self, remaining_budget, top_k=10):
        """Get popular recipes that fit the remaining calorie budget (cold-start for unknown users)"""
        recommendations = []

        # Filter recipes by remaining budget with tolerance
        budget_tolerance = max(remaining_budget * 0.3, 500)  # Allow 30% variance or 500 cal min
        filtered = self.recipes_df[
            (self.recipes_df['calories'] > 100) &  # Skip very low calorie recipes
            (self.recipes_df['calories'] <= (remaining_budget + budget_tolerance))
        ].copy()

        if len(filtered) == 0:
            # If no perfect matches, get whatever fits without going too far over
            filtered = self.recipes_df[
                (self.recipes_df['calories'] > 100) &
                (self.recipes_df['calories'] <= (remaining_budget + remaining_budget * 0.5))
            ].copy()

        if len(filtered) > 0:
            # Sort by number of reviews (popularity) descending, then by calories match
            filtered['popularity'] = filtered.get('n_steps', 0) * filtered.get('n_ingredients', 0)
            filtered['cal_diff'] = abs(filtered['calories'] - remaining_budget)
            filtered = filtered.sort_values(['popularity', 'cal_diff'], ascending=[False, True])





            # Return top_k recipes
            for idx, row in filtered.head(top_k).iterrows():
                info = row.to_dict()
                info['fits_budget'] = row['calories'] <= (remaining_budget + 100)
                info['score'] = 0.5  # Default score for cold-start
                recommendations.append(info)

        return recommendations

    def recommend_with_context(self, user_id, food_name, current_calories,
                              daily_goal=2000, top_k=10, context_weight=0.5):
        """
        Recommend recipes based on:
        1. Uploaded food context (similarity to detected food)
        2. User preferences (if known user)
        3. Calorie budget fit

        Args:
            user_id: User identifier
            food_name: Detected food name from image classifier
            current_calories: Calories consumed so far
            daily_goal: Daily calorie target
            top_k: Number of recommendations
            context_weight: Weight for context similarity (0-1)
                           Higher = more similar to uploaded food
                           Lower = more personalized to user history

        Returns:
            List of recommended recipes with metadata
        """
        # Ensure user_id is a string for consistent lookup
        user_id = str(user_id)
        remaining_budget = daily_goal - current_calories

        # 1. Find anchor recipe matching food name
        print(f"Step 1/5: Finding anchor recipe for food '{food_name}'...")
        anchor_idx = self._find_anchor_recipe(food_name)
        anchor_emb = self.item_embeddings[anchor_idx].unsqueeze(0)
        print(f"  Found anchor recipe at index {anchor_idx}")

        # 2. Compute context similarity scores (how similar recipes are to uploaded food)
        print(f"Step 2/5: Computing similarity with {len(self.item_embeddings)} recipes...")
        with torch.no_grad():
            context_scores = torch.matmul(anchor_emb, self.item_embeddings.T).squeeze(0)
        print("  Similarity computation complete")
        
        # Normalize context scores to [0, 1] for fair blending
        context_scores_min = context_scores.min()
        context_scores_max = context_scores.max()
        if context_scores_max > context_scores_min:
            context_scores = (context_scores - context_scores_min) / (context_scores_max - context_scores_min)
        else:
            context_scores = torch.ones_like(context_scores)

        # 3. Get user preference scores (if known user)
        print(f"Step 3/5: Checking user preferences...")

            # Check if user is a demo user (added at runtime, not in trained model)
        is_demo_user = hasattr(self, 'demo_users') and user_id in self.demo_users

        if is_demo_user:
            # Demo user: use history-based scoring (skip user ID embedding)
            print(f"  Demo user '{user_id}' detected (not in trained model)")
            if user_id in self.user_history and len(self.user_history[user_id]) > 0:
                # Get history embeddings
                history_indices = self.user_history[user_id]
                history_embs = self.item_embeddings[history_indices]  # [H, D]

                # Average history embeddings to get user preference
                user_pref_emb = history_embs.mean(dim=0, keepdim=True)  # [1, D]
                user_scores = torch.matmul(user_pref_emb, self.item_embeddings.T).squeeze(0)
                
                # Normalize user scores to [0, 1]
                user_scores_min = user_scores.min()
                user_scores_max = user_scores.max()
                if user_scores_max > user_scores_min:
                    user_scores = (user_scores - user_scores_min) / (user_scores_max - user_scores_min)
                else:
                    user_scores = torch.ones_like(user_scores)

                # Blend history and current food context
                # Favor the current food to change results when uploads differ
                user_weight = 0.4
                context_weight_adjusted = 0.6
                scores = user_weight * user_scores + context_weight_adjusted * context_scores
                print(f"  Using {len(history_indices)} interactions + food context (weight: {user_weight:.1f} user, {context_weight_adjusted:.1f} context)")
            else:
                # Demo user with no history: use context only
                scores = context_scores
                print("  No interaction history, using food context only")

        elif user_id in self.user_to_idx:
            # Trained user: use full model (user ID + history)
            user_idx = self.user_to_idx[user_id]
            user_tensor = torch.tensor([user_idx], device=self.device)
            history_embs, history_mask = self._get_user_history_embeddings(user_id)

            with torch.no_grad():
                user_emb = self.model.get_user_embedding(user_tensor, history_embs, history_mask)
                user_scores = torch.matmul(user_emb, self.item_embeddings.T).squeeze(0)

            # Blend user and context scores
            scores = (1 - context_weight) * user_scores + context_weight * context_scores
            print(f"  Known user '{user_id}' (trained model), blending preferences ({1-context_weight:.1f}) with food context ({context_weight:.1f})")
        else:
            # Unknown user: use only context scores (food similarity)
            scores = context_scores
            sample_users = list(self.user_to_idx.keys())[:5]
            print(f"  Unknown user '{user_id}' (type: {type(user_id).__name__})")
            print(f"  Available sample users: {sample_users}")
            print(f"  Total users in model: {len(self.user_to_idx)}")
            print("  Using food-context-based recommendations (no user history)")

        # 4. Apply calorie budget re-ranking (optimized with vectorization where possible)
        # Boost recipes that fit the remaining daily calorie budget
        print(f"Step 4/5: Applying calorie budget ranking (remaining: {remaining_budget} cal)...")

        # Optimize: Only boost top candidates to avoid iterating through all recipes
        # First get top candidates based on context/user scores
        top_candidate_indices = torch.topk(scores, k=min(top_k*10, len(scores))).indices.cpu().numpy()

        for idx in top_candidate_indices:
            recipe_id = str(self.idx_to_recipe[int(idx)])  # Ensure proper type conversion
            matching_rows = self.recipes_df[self.recipes_df['recipe_id'] == recipe_id]

            if len(matching_rows) > 0:
                calories = float(matching_rows.iloc[0]['calories'])
                if remaining_budget > 0:
                    # Calculate budget fitness: 1.0 when perfect match, decreases with difference
                    budget_fit = max(0, 1 - abs(calories - remaining_budget) / max(remaining_budget, 500))
                    # Apply 0.2 weight to budget fit (less than context/user to prioritize food relevance)
                    scores[int(idx)] = scores[int(idx)] + 0.2 * budget_fit

        # 5. Get final top-k candidates (re-rank after budget adjustment)
        print(f"Step 5/5: Selecting top {top_k} recommendations...")
        top_indices = torch.topk(scores, k=min(top_k*3, len(scores))).indices.cpu().numpy()
        print(f"  Found {len(top_indices)} candidates after scoring")
        
        # Debug: Show top 3 scores before extraction
        top_3_scores = scores[torch.topk(scores, k=min(3, len(scores))).indices]
        print(f"  Top 3 scores: {[float(s.cpu().item()) for s in top_3_scores]}")

        recommendations = []
        for i, idx in enumerate(top_indices):
            try:
                recipe_id = str(self.idx_to_recipe[int(idx)])  # Ensure proper type conversion
                matching_rows = self.recipes_df[self.recipes_df['recipe_id'] == recipe_id]
                if len(matching_rows) == 0:
                    continue
                row = matching_rows.iloc[0]
                
                info = {}
                # Manually build dict to avoid pandas Series issues
                for key in row.index:
                    val = row[key]
                    # Skip array/list columns that aren't JSON serializable
                    if isinstance(val, np.ndarray):
                        info[key] = val.tolist()
                    elif isinstance(val, list):
                        info[key] = val
                    else:
                        # Use numpy's isnan for scalar values
                        try:
                            if pd.isna(val):
                                info[key] = None
                            elif isinstance(val, (np.integer, np.floating)):
                                info[key] = float(val) if isinstance(val, np.floating) else int(val)
                            else:
                                info[key] = val
                        except (ValueError, TypeError):
                            # If isna fails, just convert to string
                            info[key] = str(val)

                # Safely check calorie budget (handle None and convert to scalar)
                cal_value = info.get('calories')
                info['fits_budget'] = bool(cal_value is not None and float(cal_value) <= (remaining_budget + 100))
                info['score'] = float(scores[int(idx)].cpu().item())
                # Add context relevance indicator
                context_score = float(context_scores[int(idx)].cpu().item())
                info['context_match'] = 'high' if context_score > 0.5 else 'medium'

                recommendations.append(info)
                if len(recommendations) >= top_k:
                    break
            except Exception as e:
                print(f"  Warning: Error processing recommendation {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"✓ Generated {len(recommendations)} recommendations successfully")
        return recommendations
