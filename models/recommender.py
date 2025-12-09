import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------
# Preprocessing Utilities
# -------------------------------------------------------------------
def parse_list_column(val):
    """Safely parses stringified lists like \"['a', 'b']\" or returns empty list."""
    try:
        return ast.literal_eval(val)
    except:
        return []

def get_ingredient_vocab(recipes_df, top_k=2000):
    """
    Builds a vocabulary of the top_k most frequent ingredients.
    Returns:
        vocab (dict): {ingredient_name: id}
        unk_idx (int): Index for unknown ingredients
    """
    all_ingredients = []
    for ing_list in recipes_df['ingredients']:
        # ing_list is already parsed
        all_ingredients.extend(ing_list)
        
    counts = Counter(all_ingredients)
    common = counts.most_common(top_k)
    
    # 0 is padding, 1 is existing...
    # Let's map directly:
    vocab = {ing: i+1 for i, (ing, _) in enumerate(common)} # 1-based index
    unk_idx = 0 # 0 for unknown/padding
    
    print(f"Vocab built with {len(vocab)} ingredients (Top {top_k}).")
    return vocab, unk_idx

# -------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------
class RecipeDataset(Dataset):
    """
    Dataset for serving Recipe Features (ID, Ingredients, Nutrition).
    """
    def __init__(self, recipes_csv, vocab=None):
        self.recipes = pd.read_csv(recipes_csv)
        
        # 1. ID Handling
        if "id" in self.recipes.columns and "recipe_id" not in self.recipes.columns:
            self.recipes = self.recipes.rename(columns={"id": "recipe_id"})
        self.recipes["recipe_id"] = self.recipes["recipe_id"].astype(str)
        self.unique_recipe_ids = self.recipes["recipe_id"].unique()
        
        # 2. Parsing Content
        print("Parsing ingredients and nutrition...")
        self.recipes['ingredients'] = self.recipes['ingredients'].apply(parse_list_column)
        self.recipes['nutrition'] = self.recipes['nutrition'].apply(parse_list_column)
        
        # 3. Build Vocab if not provided
        if vocab is None:
            self.vocab, self.unk_idx = get_ingredient_vocab(self.recipes)
        else:
            self.vocab = vocab
            self.unk_idx = 0
            
        # 4. Pre-process features into tensors to save time during training
        self.feature_map = {}
        
        for idx, row in self.recipes.iterrows():
            rid = row['recipe_id']
            
            # Ingredients -> Indices
            ing_indices = [self.vocab.get(ing, self.unk_idx) for ing in row['ingredients']]
            if not ing_indices:
                ing_indices = [self.unk_idx]
            
            # Nutrition -> Float Tensor (Normalize roughly)
            # Standard format: [Cal, Fat, Sugar, Sodium, Protein, SatFat, Carbs]
            # Simple scaling to keep them somewhat small (e.g. div max or standard)
            # For simplicity & robustness: log1p or simple division
            nut = np.array(row['nutrition'], dtype=np.float32)
            # Avoid nan
            nut = np.nan_to_num(nut)
            # Simple manual normalization based on typical food ranges
            # Cals/1000, Fat/100, Sugar/100, Sod/1000, Prot/100, SatFat/100, Carb/100
            scale = np.array([1000, 100, 100, 1000, 100, 100, 100], dtype=np.float32)
            # Safety check for lengths
            if len(nut) == 7:
                nut = nut / scale
            else:
                nut = np.zeros(7, dtype=np.float32)
                
            self.feature_map[rid] = {
                'ingredients': torch.tensor(ing_indices, dtype=torch.long),
                'nutrition': torch.tensor(nut, dtype=torch.float32)
            }

        # ID Mappings
        self.recipe_to_idx = {rid: i for i, rid in enumerate(self.unique_recipe_ids)}
        self.idx_to_recipe = {i: rid for rid, i in self.recipe_to_idx.items()}
        
    def __len__(self):
        return len(self.unique_recipe_ids)
        
    def __getitem__(self, idx):
        # Return ID index 
        # (Content features accessed via lookups in Collate or Model if needed, 
        # but standard pattern is to return everything. Let's return everything.)
        rid = self.idx_to_recipe[idx]
        feats = self.feature_map[rid]
        
        return {
            'recipe_idx': torch.tensor(idx, dtype=torch.long),
            'ingredients': feats['ingredients'], 
            'nutrition': feats['nutrition']
        }

class InteractionDataset(Dataset):
    def __init__(self, interactions_csv, users_csv, recipe_dataset):
        self.interactions = pd.read_csv(interactions_csv)
        self.recipe_dataset = recipe_dataset
        
        self.interactions["user_id"] = self.interactions["user_id"].astype(str)
        self.interactions["recipe_id"] = self.interactions["recipe_id"].astype(str)
        
        # Filter
        known_recipes = set(self.recipe_dataset.recipe_to_idx.keys())
        self.interactions = self.interactions[self.interactions["recipe_id"].isin(known_recipes)]
        
        # User Mappings
        self.unique_user_ids = self.interactions["user_id"].unique()
        self.user_to_idx = {uid: i for i, uid in enumerate(self.unique_user_ids)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        
    def __len__(self):
        return len(self.interactions)
        
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        uid = row["user_id"]
        rid = row["recipe_id"]
        
        user_idx = self.user_to_idx[uid]
        # Get recipe features directly from the recipe_dataset using the ID
        recipe_mem_idx = self.recipe_dataset.recipe_to_idx[rid]
        recipe_data = self.recipe_dataset[recipe_mem_idx]
        
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'recipe_idx': recipe_data['recipe_idx'],
            'ingredients': recipe_data['ingredients'],
            'nutrition': recipe_data['nutrition']
        }

def collate_fn(batch):
    """
    Custom collate because ingredients are variable length lists.
    We need to pad them.
    """
    user_idxs = torch.stack([x['user_idx'] for x in batch])
    recipe_idxs = torch.stack([x['recipe_idx'] for x in batch])
    nutrition = torch.stack([x['nutrition'] for x in batch])
    
    # Pad ingredients (EmbeddingBag can take offsets, or we can just pad with 0)
    # Using EmbeddingBag with offsets is most efficient, but padding is easier to debug.
    # Let's use padding.
    ingredients_list = [x['ingredients'] for x in batch]
    max_len = max([len(i) for i in ingredients_list])
    
    padded_ingredients = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(ingredients_list):
        end = len(seq)
        padded_ingredients[i, :end] = seq
        
    return {
        'user_idx': user_idxs,
        'recipe_idx': recipe_idxs,
        'ingredients': padded_ingredients,
        'nutrition': nutrition
    }

# -------------------------------------------------------------------
# Model with Content Features
# -------------------------------------------------------------------
class RecipeTower(nn.Module):
    def __init__(self, num_recipes, vocab_size, embedding_dim=32):
        super().__init__()
        # 1. ID Embedding
        self.id_embedding = nn.Embedding(num_recipes, embedding_dim)
        
        # 2. Ingredients Embedding (Bag of Words style)
        # vocab_size+1 because 0 is unk/padding
        self.ingredient_embedding = nn.EmbeddingBag(vocab_size + 1, embedding_dim, mode='mean')
        
        # 3. Nutrition Dense
        self.nutrition_dense = nn.Linear(7, embedding_dim)
        
        # 4. Fusion
        # Concatenate 3 vectors -> Dense -> Output
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim) # Project back to common space
        )
        
    def forward(self, recipe_indices, ingredient_indices, nutrition_tensor):
        id_emb = self.id_embedding(recipe_indices) # [B, Dim]
        
        # EmbeddingBag expects 2D input for padding mode if separate? 
        # Actually EmbeddingBag(input, offsets) is 1D. 
        # EmbeddingBag(input) where input is 2D handles padding if padding_idx is set.
        # But we used mode='mean' and 0 is unk. 
        ing_emb = self.ingredient_embedding(ingredient_indices) # [B, Dim]
        
        nut_feat = self.nutrition_dense(nutrition_tensor) # [B, Dim]
        
        # Concatenate
        combined = torch.cat([id_emb, ing_emb, nut_feat], dim=1) # [B, Dim*3]
        return self.fusion(combined)

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_recipes, vocab_size, embedding_dim=32):
        super().__init__()
        # User Tower (Simple ID for now, could add history LSTM later)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Recipe Tower (Complex)
        self.recipe_tower = RecipeTower(num_recipes, vocab_size, embedding_dim)
        
    def forward(self, user_indices, recipe_indices, ingredients, nutrition):
        user_emb = self.user_embedding(user_indices)
        recipe_emb = self.recipe_tower(recipe_indices, ingredients, nutrition)
        return user_emb, recipe_emb
    
    def get_user_embedding(self, user_indices):
        return self.user_embedding(user_indices)
        
    # Helper to get recipe embedding from raw features
    def get_recipe_embedding(self, recipe_indices, ingredients, nutrition):
        return self.recipe_tower(recipe_indices, ingredients, nutrition)
