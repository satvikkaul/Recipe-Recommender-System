import torch
import pandas as pd
import numpy as np
import time
import math
import sys
import os
from tqdm import tqdm

# Add current directory to path so we can import backend/models
sys.path.append(os.getcwd())
from backend.inference import RecommenderEngine

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_DIR = "data/food.com-interaction"
RECIPES_CSV = f"{DATA_DIR}/RAW_recipes.csv"
INTERACTIONS_CSV = f"{DATA_DIR}/RAW_interactions.csv"
MODEL_PATH = "models/saved/recommender_model_pytorch.pth"
SAMPLE_SIZE = 5000 
TOP_K = 10

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def calculate_metrics(recommended_ids, true_id):
    k = len(recommended_ids)
    if k == 0: return 0.0, 0.0, 0.0
    
    if true_id in recommended_ids:
        p_k = 1.0 / k
        r_k = 1.0
        rank_index = recommended_ids.index(true_id)
        dcg = 1.0 / math.log2(rank_index + 2)
        ndcg_k = dcg
    else:
        p_k = 0.0
        r_k = 0.0
        ndcg_k = 0.0
        
    return p_k, r_k, ndcg_k

def evaluate():
    print("=== Two-Tower Model Evaluation ===")
    
    # 1. Load Engine (Loads Model + Precomputes Embeddings)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Initialize Engine
    # This handles loading the vocabulary, model weights, and precomputing recipe embeddings
    engine = RecommenderEngine(MODEL_PATH, RECIPES_CSV)
    
    # 2. Load Interactions for Testing
    print("Loading Interactions...")
    interactions = pd.read_csv(INTERACTIONS_CSV)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["recipe_id"] = interactions["recipe_id"].astype(str)
    
    # Filter to known items (in case model vocab implies some missing)
    known_ids = set(engine.recipe_to_idx.keys())
    interactions = interactions[interactions["recipe_id"].isin(known_ids)]
    
    # 3. Filter Warm Users (Same as Baseline)
    user_counts = interactions['user_id'].value_counts()
    warm_users = user_counts[user_counts >= 5].index.tolist()
    print(f"Filtering for Warm Users (History >= 5)...")
    warm_interactions = interactions[interactions['user_id'].isin(warm_users)]
    
    # 4. Sample
    actual_sample_size = min(SAMPLE_SIZE, len(warm_interactions))
    eval_set = warm_interactions.sample(n=actual_sample_size, random_state=42)
    
    print(f"Evaluating on {actual_sample_size} samples...")
    print("-" * 70)
    
    # Metrics
    metrics = {"p": 0.0, "r": 0.0, "n": 0.0}
    results_data = [] # NEW: Collection for CSV export
    count = 0
    skipped = 0
    
    # For masking history (we don't want to recommend what they already saw... 
    # OR maybe we do if we just want raw ranking? 
    # Usually we mask history. The engine doesn't automatically mask history 
    # because it doesn't store user history state, it just ranks.)
    user_history = interactions.groupby("user_id")["recipe_id"].apply(set).to_dict()
    
    loop = tqdm(eval_set.iterrows(), total=actual_sample_size, unit="user")
    
    # Access internals for speed
    model = engine.model
    device = engine.device
    item_embs = engine.item_embeddings # [N_Items, Dim]
    
    for idx, row in loop:
        uid = row['user_id']
        true_rid = row['recipe_id']
        
        if uid not in engine.user_to_idx:
            skipped += 1
            continue
            
        # Get User Embedding
        u_idx = engine.user_to_idx[uid]
        u_tensor = torch.tensor([u_idx], device=device)
        
        with torch.no_grad():
            u_emb = model.get_user_embedding(u_tensor) # [1, Dim]
            
        # Score all items (Dot Product)
        # item_embs is [N, Dim], u_emb is [1, Dim]
        # scores: [1, N]
        scores = torch.matmul(u_emb, item_embs.T).squeeze(0)
        
        # We need to find the rank of 'true_rid'
        # To do this correctly with masking:
        # 1. Mask out history (except true_rid, which is temporarily 'future')
        # Actually, standard Eval is: Given History H, predict True next item T.
        # So we should mask H (excluding T).
        
        history = user_history.get(uid, set())
        # Mask indices
        # We need to convert ID -> Index for masking
        mask_indices = []
        for h_rid in history:
            if h_rid == true_rid: continue
            if h_rid in engine.recipe_to_idx:
                mask_indices.append(engine.recipe_to_idx[h_rid])
        
        if mask_indices:
            scores[mask_indices] = -float('inf')
            
        # Top K
        top_k_vals, top_k_inds = torch.topk(scores, TOP_K)
        top_k_inds = top_k_inds.cpu().numpy()
        
        # Convert to IDs
        rec_ids = [engine.idx_to_recipe[i] for i in top_k_inds]
        
        # Calc Metrics
        p, r, n = calculate_metrics(rec_ids, true_rid)
        
        # Store for CSV
        results_data.append({
            "user_id": uid,
            "recipe_id": true_rid,
            "ndcg_10": n,
            "recall_10": r,
            "precision_10": p
        })
        
        metrics["p"] += p
        metrics["r"] += r
        metrics["n"] += n
        count += 1
        
        if count % 50 == 0:
            loop.set_postfix(ndcg=f"{metrics['n']/count:.4f}", recall=f"{metrics['r']/count:.4f}")
            
    print("\n" + "="*70)
    print(f"TWO-TOWER MODEL RESULTS (Warm Users, N={count})")
    print("="*70)
    if count > 0:
        print(f"  NDCG@10:      {metrics['n']/count:.5f}")
        print(f"  Recall@10:    {metrics['r']/count:.5f}")
        print(f"  Precision@10: {metrics['p']/count:.5f}")
        
        # Save CSV
        csv_path = "models/saved/two_tower_results.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(results_data).to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")
    else:
        print("No valid users evaluating.")
    print("="*70)

if __name__ == "__main__":
    evaluate()
