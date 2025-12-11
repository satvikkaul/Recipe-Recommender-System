import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import ast
import math
import time
from datetime import datetime
import json
import os

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_DIR = "data/food.com-interaction"
RECIPES_CSV = f"{DATA_DIR}/RAW_recipes.csv"
INTERACTIONS_CSV = f"{DATA_DIR}/RAW_interactions.csv"
SAMPLE_SIZE = 5000  # Evaluate on 5000 random interactions for speed
TOP_K = 10

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def calculate_metrics(recommended_ids, true_id):
    """
    Computes P@K, R@K, NDCG@K, MRR@K for a single user interaction.
    Args:
        recommended_ids (list): Top K recommended recipe IDs.
        true_id (str): The actual recipe ID the user interacted with.
    """
    k = len(recommended_ids)
    
    # Handle empty recommendations
    if k == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Hit?
    if true_id in recommended_ids:
        # Precision@K = 1/K if hit, else 0 (Simplistic for single-target)
        p_k = 1.0 / k
        # Recall@K = 1 if hit, else 0
        r_k = 1.0
        
        # NDCG@K
        rank_index = recommended_ids.index(true_id)
        dcg = 1.0 / math.log2(rank_index + 2)
        idcg = 1.0
        ndcg_k = dcg / idcg
        
        # MRR@K
        # 1 / (rank + 1)
        mrr_k = 1.0 / (rank_index + 1)
    else:
        p_k = 0.0
        r_k = 0.0
        ndcg_k = 0.0
        mrr_k = 0.0
        
    return p_k, r_k, ndcg_k, mrr_k

# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------
def load_data():
    print("Loading Data...")
    recipes = pd.read_csv(RECIPES_CSV)
    interactions = pd.read_csv(INTERACTIONS_CSV)
    
    # Standardize IDs
    if "id" in recipes.columns and "recipe_id" not in recipes.columns:
        recipes = recipes.rename(columns={"id": "recipe_id"})
    
    recipes["recipe_id"] = recipes["recipe_id"].astype(str)
    interactions["recipe_id"] = interactions["recipe_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    
    # Filter interactions to known recipes
    known_ids = set(recipes["recipe_id"])
    interactions = interactions[interactions["recipe_id"].isin(known_ids)]
    
    return recipes, interactions

# -------------------------------------------------------------------
# Baseline 1: Popularity
# -------------------------------------------------------------------
class PopularityRecommender:
    def __init__(self, interactions):
        # Count frequency of each recipe
        counts = interactions["recipe_id"].value_counts()
        self.top_items = counts.head(TOP_K).index.tolist()
        print(f"Popularity Model trained. Top item: {self.top_items[0]}")
        
    def recommend(self, user_id):
        # Always return global top items
        return self.top_items

# -------------------------------------------------------------------
# Baseline 2: Content-Based KNN
# -------------------------------------------------------------------
class ContentKNNRecommender:
    def __init__(self, recipes_df):
        print("Training KNN (TF-IDF on Ingredients)...")
        
        def clean_ing(val):
            try:
                lst = ast.literal_eval(val)
                return " ".join(lst)
            except:
                return ""
                
        self.recipes = recipes_df.copy()
        self.recipes['clean_ingredients'] = self.recipes['ingredients'].apply(clean_ing)
        
        # REDUCED VOCAB SIZE for speed (was 5000, now 1000)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.recipes['clean_ingredients'])
        
        # OPTIMIZATION: Use direct Matrix Multiplication instead of NearestNeighbors
        # This is O(N) but fast for sparse matrices.
        # Self-similarity dot product is effectively Cosine Similarity if L2 normalized.
        # TfidfVectorizer returns L2 normalized vectors by default.
        self.idx_to_id = self.recipes['recipe_id'].values
        self.id_to_idx = {rid: i for i, rid in enumerate(self.idx_to_id)}
        print("KNN Trained (Matrix ready).")
        
    def recommend_profile(self, recipe_ids):
        """
        Recommend items using 'Max-Similarity' strategy.
        Optimized with direct sparse matrix multiplication.
        """
        # 1. Get query vectors for history items
        valid_indices = [self.id_to_idx[rid] for rid in recipe_ids if rid in self.id_to_idx]
        
        if not valid_indices:
            return []
            
        # [M, Features] where M is history length
        query_vectors = self.tfidf_matrix[valid_indices]
        
        # 2. Compute similarity to ALL items
        # Result: [M, N_Recipes]
        # This might be memory heavy if M * N is huge.
        # But M is small (history length), N is 230k.
        # 10 * 230k is fine (2.3M floats = ~9MB).
        
        # We want MAX similarity across history items for each target item.
        # "Is item X similar to any item in my history?"
        
        scores_matrix = self.tfidf_matrix.dot(query_vectors.T).toarray() # [N, M]
        
        # Max across columns (history items)
        # Result: [N]
        max_scores = scores_matrix.max(axis=1)
        
        # 3. Top K
        # Use argpartition for speed O(N) instead of Sort O(N log N)
        # We want top K
        K = TOP_K
        if len(max_scores) <= K:
             top_indices = np.argsort(max_scores)[::-1]
        else:
            # Get indices of top K (unsorted)
            top_indices_unsorted = np.argpartition(max_scores, -K)[-K:]
            # Sort just these K
            top_scores = max_scores[top_indices_unsorted]
            sorted_local_indices = np.argsort(top_scores)[::-1]
            top_indices = top_indices_unsorted[sorted_local_indices]
            
        # Map back to IDs
        rec_ids = [self.idx_to_id[i] for i in top_indices]
        
        # Filter out input items? usually yes.
        # But for strictly following the previous logic, we just return top matches.
        # Let's filter out history items if they appear.
        rec_ids = [rid for rid in rec_ids if rid not in recipe_ids][:TOP_K]
        
        return rec_ids

# -------------------------------------------------------------------
# Evaluation Loop
# -------------------------------------------------------------------
def evaluate():
    recipes, all_interactions = load_data()
    
    print("Training Baselines...")
    start_train = time.time()
    pop_model = PopularityRecommender(all_interactions)
    knn_model = ContentKNNRecommender(recipes)
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.2f}s\n")
    
    # FILTER: Only evaluate "Warm" users (At least 5 interactions)
    # After removing target, they'll have at least 4 in history for KNN
    user_counts = all_interactions['user_id'].value_counts()
    warm_users = user_counts[user_counts >= 5].index.tolist()
    
    print(f"Filtering for Warm Users (History >= 5)...")
    warm_interactions = all_interactions[all_interactions['user_id'].isin(warm_users)]
    
    print(f"Evaluating on up to {SAMPLE_SIZE} random samples from Warm Users...")
    # Use min to handle case where we have fewer warm interactions than SAMPLE_SIZE
    actual_sample_size = min(SAMPLE_SIZE, len(warm_interactions))
    eval_set = warm_interactions.sample(n=actual_sample_size, random_state=42)
    
    pop_metrics = {"p": 0.0, "r": 0.0, "n": 0.0, "mrr": 0.0}
    knn_metrics = {"p": 0.0, "r": 0.0, "n": 0.0, "mrr": 0.0}
    
    pop_recommended_items = set()
    knn_recommended_items = set()
    
    user_history = all_interactions.groupby("user_id")["recipe_id"].apply(list).to_dict()
    
    from tqdm import tqdm
    print(f"\nRunning evaluation on {actual_sample_size} samples...")
    print("-" * 70)
    
    count = 0
    skipped = 0
    start_eval = time.time()
    
    # Use list() to force tqdm to know total size if iterrows doesn't provide it easily
    # But iterrows is generator.
    
    loop = tqdm(eval_set.iterrows(), total=actual_sample_size, desc="Evaluating", unit="user")
    
    for idx, row in loop:
        uid = row['user_id']
        true_rid = row['recipe_id']
        
        # Get history excluding current target
        history = user_history.get(uid, [])
        history_minus_target = [x for x in history if x != true_rid]
        
        # Skip only if completely empty (shouldn't happen with >= 5 filter)
        if len(history_minus_target) == 0:
            skipped += 1
            continue
            
        # 1. Popularity
        pop_recs = pop_model.recommend(uid)
        pop_recommended_items.update(pop_recs)
        
        p, r, n, m = calculate_metrics(pop_recs, true_rid)
        pop_metrics["p"] += p
        pop_metrics["r"] += r
        pop_metrics["n"] += n
        pop_metrics["mrr"] += m
        
        # 2. KNN (Content) - Uses all history
        knn_recs = knn_model.recommend_profile(history_minus_target)
        knn_recommended_items.update(knn_recs)
        
        p, r, n, m = calculate_metrics(knn_recs, true_rid)
        knn_metrics["p"] += p
        knn_metrics["r"] += r
        knn_metrics["n"] += n
        knn_metrics["mrr"] += m
        
        count += 1
        
        # Optional: Print live stats in postfix
        if count % 10 == 0:
             loop.set_postfix(
                 knn_ndcg=f"{knn_metrics['n']/count:.4f}", 
                 pop_ndcg=f"{pop_metrics['n']/count:.4f}"
             )
    
    print()  # New line after progress
    total_eval_time = time.time() - start_eval
    
    # Print Results
    print("-" * 70)
    print(f"Evaluation completed in {total_eval_time:.2f}s ({total_eval_time/count:.4f}s per sample)")
    print(f"Processed: {count} valid samples, Skipped: {skipped} empty histories\n")
    print("="*70)
    print(f"BASELINE RESULTS (Warm Users, N={count})")
    print("="*70)
    
    if count > 0:
        # Calculate Coverage
        total_unique_items = len(recipes)
        pop_coverage = len(pop_recommended_items) / total_unique_items
        knn_coverage = len(knn_recommended_items) / total_unique_items
        
        print(f"\nPOPULARITY BASELINE:")
        print(f"  NDCG@10:      {pop_metrics['n']/count:.5f}")
        print(f"  Recall@10:    {pop_metrics['r']/count:.5f}")
        print(f"  Precision@10: {pop_metrics['p']/count:.5f}")
        print(f"  MRR@10:       {pop_metrics['mrr']/count:.5f}")
        print(f"  Coverage:     {pop_coverage:.2%}")
        
        print("\n" + "-" * 70)  
        print(f"\nKNN (CONTENT) BASELINE:")
        print(f"  NDCG@10:      {knn_metrics['n']/count:.5f}")
        print(f"  Recall@10:    {knn_metrics['r']/count:.5f}")
        print(f"  Precision@10: {knn_metrics['p']/count:.5f}")
        print(f"  MRR@10:       {knn_metrics['mrr']/count:.5f}")
        print(f"  Coverage:     {knn_coverage:.2%}")
        
        # Compare
        print("\n" + "-" * 70)
        print(f"\nComparison (KNN vs Popularity):")
        knn_ndcg_gain = (knn_metrics['n'] - pop_metrics['n'])/count
        knn_recall_gain = (knn_metrics['r'] - pop_metrics['r'])/count
        knn_precision_gain = (knn_metrics['p'] - pop_metrics['p'])/count
        
        pop_ndcg = pop_metrics['n']/count
        pop_recall = pop_metrics['r']/count
        print(f"  NDCG Improvement: {knn_ndcg_gain:+.5f} ({knn_ndcg_gain/pop_ndcg*100 if pop_ndcg > 0 else 0:+.1f}%)")
        print(f"  Recall Improvement: {knn_recall_gain:+.5f} ({knn_recall_gain/pop_recall*100 if pop_recall > 0 else 0:+.1f}%)")
        
        # Save evaluation results to JSON
        eval_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_config": {
                "sample_size": SAMPLE_SIZE,
                "top_k": TOP_K,
                "warm_user_minimum_history": 5
            },
            "dataset_statistics": {
                "total_interactions": len(all_interactions),
                "warm_users_count": len(warm_users),
                "evaluation_samples": count,
                "skipped_samples": skipped
            },
            "baseline_results": {
                "popularity": {
                    "precision_10": round(pop_metrics['p']/count, 5),
                    "recall_10": round(pop_metrics['r']/count, 5),
                    "ndcg_10": round(pop_metrics['n']/count, 5),
                    "mrr_10": round(pop_metrics['mrr']/count, 5),
                    "coverage": round(pop_coverage, 5)
                },
                "content_knn": {
                    "precision_10": round(knn_metrics['p']/count, 5),
                    "recall_10": round(knn_metrics['r']/count, 5),
                    "ndcg_10": round(knn_metrics['n']/count, 5),
                    "mrr_10": round(knn_metrics['mrr']/count, 5),
                    "coverage": round(knn_coverage, 5)
                }
            },
            "knn_improvement": {
                "precision_gain": round(knn_precision_gain, 5),
                "recall_gain": round(knn_recall_gain, 5),
                "ndcg_gain": round(knn_ndcg_gain, 5),
                "ndcg_percentage_improvement": round(knn_ndcg_gain/pop_ndcg*100 if pop_ndcg > 0 else 0, 2)
            },
            "performance": {
                "total_evaluation_time_seconds": round(total_eval_time, 2),
                "time_per_sample_milliseconds": round(total_eval_time/count*1000, 3)
            }
        }
        
        # Save to JSON
        results_path = "models/saved/baseline_evaluation_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=4)
        print(f"\nEvaluation results saved to {results_path}")
    else:
        print("No valid samples found.")
    print("="*70)

if __name__ == "__main__":
    evaluate()
