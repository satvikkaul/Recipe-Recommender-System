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
    def __init__(self, interactions_csv, users_csv, recipe_dataset, max_history=15,
                 split='train', train_ratio=0.8):
        """
        Args:
            interactions_csv: Path to interactions CSV
            users_csv: Path to users CSV (unused but kept for compatibility)
            recipe_dataset: RecipeDataset instance
            max_history: Maximum number of historical interactions to use
            split: 'train', 'test', or 'all'
            train_ratio: Fraction of data to use for training (temporal split)
        """
        self.recipe_dataset = recipe_dataset
        self.max_history = max_history
        self.split = split

        # Load all interactions
        all_interactions = pd.read_csv(interactions_csv)
        all_interactions["user_id"] = all_interactions["user_id"].astype(str)
        all_interactions["recipe_id"] = all_interactions["recipe_id"].astype(str)

        # Filter to known recipes
        known_recipes = set(self.recipe_dataset.recipe_to_idx.keys())
        all_interactions = all_interactions[all_interactions["recipe_id"].isin(known_recipes)]

        # Sort by date (critical for temporal split)
        if 'date' in all_interactions.columns:
            all_interactions = all_interactions.sort_values('date').reset_index(drop=True)
        else:
            # If no date, assume order in file is chronological
            all_interactions = all_interactions.reset_index(drop=True)
            print("Warning: No 'date' column found. Assuming chronological order.")

        # Perform temporal split
        if split in ['train', 'test']:
            split_idx = int(len(all_interactions) * train_ratio)

            if split == 'train':
                self.interactions = all_interactions.iloc[:split_idx].reset_index(drop=True)
                print(f"Train split: {len(self.interactions)} interactions (first {train_ratio*100:.0f}%)")
            else:  # test
                self.interactions = all_interactions.iloc[split_idx:].reset_index(drop=True)
                # Store split index to access full history
                self.split_idx = split_idx
                # Keep reference to all interactions for history lookup
                self.all_interactions = all_interactions
                print(f"Test split: {len(self.interactions)} interactions (last {(1-train_ratio)*100:.0f}%)")
        else:  # all
            self.interactions = all_interactions
            print(f"Using all {len(self.interactions)} interactions")

        # User Mappings (from train data only for consistency)
        if split == 'train' or split == 'all':
            self.unique_user_ids = self.interactions["user_id"].unique()
        else:  # test - use train users to maintain consistency
            # For test, we still need to know which users are valid
            # We'll filter to warm-start users during evaluation
            self.unique_user_ids = self.interactions["user_id"].unique()

        self.user_to_idx = {uid: i for i, uid in enumerate(self.unique_user_ids)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}

        # Build user history map
        print("Building user history map...")
        self.user_history = {}

        if split == 'test':
            # For test split, use ALL past interactions (including train) as history
            for idx, row in self.all_interactions.iterrows():
                uid = row["user_id"]
                rid = row["recipe_id"]
                if uid not in self.user_history:
                    self.user_history[uid] = []
                self.user_history[uid].append((idx, rid))
        else:
            # For train/all, use only interactions up to current point
            for idx, row in self.interactions.iterrows():
                uid = row["user_id"]
                rid = row["recipe_id"]
                if uid not in self.user_history:
                    self.user_history[uid] = []
                self.user_history[uid].append((idx, rid))

        print(f"User history built for {len(self.user_history)} users.")

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

        # Get user's history BEFORE this interaction (temporal split)
        user_history_list = self.user_history.get(uid, [])

        # For test split, we need to map idx correctly
        if self.split == 'test':
            # Current interaction's index in the full dataset
            current_idx_in_all = self.split_idx + idx
            # Find all interactions before current one in the full dataset
            history_items = [
                (int_idx, hist_rid)
                for int_idx, hist_rid in user_history_list
                if int_idx < current_idx_in_all  # Only past interactions
            ]
        else:
            # For train/all, idx is already correct
            history_items = [
                (int_idx, hist_rid)
                for int_idx, hist_rid in user_history_list
                if int_idx < idx  # Only past interactions
            ]

        # Take the most recent max_history items
        history_items = history_items[-self.max_history:]

        # Fetch recipe features for history
        history_recipe_indices = []
        history_ingredients = []
        history_nutrition = []

        for _, hist_rid in history_items:
            if hist_rid in self.recipe_dataset.recipe_to_idx:
                hist_recipe_idx = self.recipe_dataset.recipe_to_idx[hist_rid]
                hist_data = self.recipe_dataset[hist_recipe_idx]

                history_recipe_indices.append(hist_data['recipe_idx'])
                history_ingredients.append(hist_data['ingredients'])
                history_nutrition.append(hist_data['nutrition'])

        # Convert to tensors (will be padded in collate_fn)
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'recipe_idx': recipe_data['recipe_idx'],
            'ingredients': recipe_data['ingredients'],
            'nutrition': recipe_data['nutrition'],
            'history_recipe_indices': history_recipe_indices,  # List of tensors
            'history_ingredients': history_ingredients,  # List of tensors
            'history_nutrition': history_nutrition  # List of tensors
        }

def collate_fn(batch):
    """
    Custom collate because ingredients are variable length lists and user histories vary.
    Handles:
    - Variable length ingredient lists
    - Variable length user history sequences
    - Padding and masking
    """
    user_idxs = torch.stack([x['user_idx'] for x in batch])
    recipe_idxs = torch.stack([x['recipe_idx'] for x in batch])
    nutrition = torch.stack([x['nutrition'] for x in batch])

    # Pad target recipe ingredients
    ingredients_list = [x['ingredients'] for x in batch]
    max_ing_len = max([len(i) for i in ingredients_list]) if ingredients_list else 1

    padded_ingredients = torch.zeros((len(batch), max_ing_len), dtype=torch.long)
    for i, seq in enumerate(ingredients_list):
        end = len(seq)
        padded_ingredients[i, :end] = seq

    # Process history data
    # Each batch item has a list of history items (variable length)
    batch_size = len(batch)
    history_lengths = [len(x['history_recipe_indices']) for x in batch]
    max_history_len = max(history_lengths) if history_lengths else 1

    # Handle empty history case
    if max_history_len == 0:
        max_history_len = 1  # Prevent zero-sized tensors

    # Initialize history tensors
    history_recipe_indices = torch.zeros((batch_size, max_history_len), dtype=torch.long)
    history_mask = torch.zeros((batch_size, max_history_len), dtype=torch.bool)

    # For ingredients, we need [batch, history_len, max_ing_len]
    # First, find max ingredient length across all history items
    all_history_ings = []
    for x in batch:
        all_history_ings.extend(x['history_ingredients'])

    max_hist_ing_len = max([len(ing) for ing in all_history_ings]) if all_history_ings else 1

    history_ingredients = torch.zeros((batch_size, max_history_len, max_hist_ing_len), dtype=torch.long)
    history_nutrition = torch.zeros((batch_size, max_history_len, 7), dtype=torch.float32)

    # Fill history tensors
    for i, item in enumerate(batch):
        hist_len = len(item['history_recipe_indices'])

        if hist_len > 0:
            # Recipe indices
            for j in range(hist_len):
                history_recipe_indices[i, j] = item['history_recipe_indices'][j]
                history_mask[i, j] = True

                # Ingredients
                hist_ing = item['history_ingredients'][j]
                ing_len = len(hist_ing)
                history_ingredients[i, j, :ing_len] = hist_ing

                # Nutrition
                history_nutrition[i, j] = item['history_nutrition'][j]

    return {
        'user_idx': user_idxs,
        'recipe_idx': recipe_idxs,
        'ingredients': padded_ingredients,
        'nutrition': nutrition,
        'history_recipe_indices': history_recipe_indices,
        'history_ingredients': history_ingredients,
        'history_nutrition': history_nutrition,
        'history_mask': history_mask
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

class UserTower(nn.Module):
    """
    User Tower that combines:
    1. User ID embedding (for personalization & cold-start)
    2. Interaction history encoding via GRU (for collaborative filtering)
    """
    def __init__(self, num_users, embedding_dim=32, history_encoder_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim

        # User ID embedding
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)

        # History encoder: takes sequence of recipe embeddings
        # Input: [batch, seq_len, embedding_dim]
        # Output: [batch, hidden_dim]
        self.history_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=history_encoder_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Fusion layer to combine user_id_emb + history_emb
        # Using addition requires same dim, or we can use a small MLP
        if history_encoder_dim != embedding_dim:
            self.fusion = nn.Linear(history_encoder_dim, embedding_dim)
        else:
            self.fusion = None

    def forward(self, user_indices, history_embeddings=None, history_mask=None):
        """
        Args:
            user_indices: [B] tensor of user IDs
            history_embeddings: [B, seq_len, embedding_dim] tensor of recipe embeddings from history
            history_mask: [B, seq_len] boolean mask (True = valid, False = padding)

        Returns:
            user_emb: [B, embedding_dim] combined user representation
        """
        # Get base user embedding
        user_id_emb = self.user_id_embedding(user_indices)  # [B, D]

        # If no history provided, return only user ID embedding
        if history_embeddings is None or history_embeddings.size(1) == 0:
            return user_id_emb

        # Encode history with GRU
        # GRU returns: output [B, seq_len, hidden], hidden [1, B, hidden]
        _, history_hidden = self.history_encoder(history_embeddings)
        history_emb = history_hidden.squeeze(0)  # [B, hidden_dim]

        # Fuse history with user ID
        if self.fusion is not None:
            history_emb = self.fusion(history_emb)  # [B, embedding_dim]

        # Combine via addition (residual connection)
        # This allows the model to learn how much to weight ID vs history
        combined_emb = user_id_emb + history_emb

        return combined_emb

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_recipes, vocab_size, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim

        # User Tower (with history encoding)
        self.user_tower = UserTower(num_users, embedding_dim, history_encoder_dim=embedding_dim)

        # Recipe Tower (Complex)
        self.recipe_tower = RecipeTower(num_recipes, vocab_size, embedding_dim)

    def forward(self, user_indices, recipe_indices, ingredients, nutrition,
                history_recipe_indices=None, history_ingredients=None,
                history_nutrition=None, history_mask=None):
        """
        Args:
            user_indices: [B] user IDs
            recipe_indices: [B] target recipe IDs
            ingredients: [B, max_ing_len] ingredient indices
            nutrition: [B, 7] nutrition features
            history_recipe_indices: [B, H] historical recipe IDs (optional)
            history_ingredients: [B, H, max_ing_len] historical ingredients (optional)
            history_nutrition: [B, H, 7] historical nutrition (optional)
            history_mask: [B, H] mask for valid history items (optional)
        """
        # Get target recipe embedding
        recipe_emb = self.recipe_tower(recipe_indices, ingredients, nutrition)

        # Get history embeddings if provided
        history_embeddings = None
        if history_recipe_indices is not None:
            batch_size, history_len = history_recipe_indices.shape

            # Flatten history to process through recipe tower
            # [B, H] -> [B*H]
            flat_recipe_idx = history_recipe_indices.reshape(-1)
            flat_ingredients = history_ingredients.reshape(batch_size * history_len, -1)
            flat_nutrition = history_nutrition.reshape(batch_size * history_len, -1)

            # Get embeddings for all history items
            flat_hist_emb = self.recipe_tower(flat_recipe_idx, flat_ingredients, flat_nutrition)

            # Reshape back to [B, H, D]
            history_embeddings = flat_hist_emb.reshape(batch_size, history_len, self.embedding_dim)

        # Get user embedding (with history encoding)
        user_emb = self.user_tower(user_indices, history_embeddings, history_mask)

        return user_emb, recipe_emb

    def get_user_embedding(self, user_indices, history_embeddings=None, history_mask=None):
        """Get user embedding with optional history"""
        return self.user_tower(user_indices, history_embeddings, history_mask)

    def get_recipe_embedding(self, recipe_indices, ingredients, nutrition):
        """Get recipe embedding from features"""
        return self.recipe_tower(recipe_indices, ingredients, nutrition)
