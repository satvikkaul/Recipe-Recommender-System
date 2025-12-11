"""
Comprehensive evaluation script for the improved Two-Tower model.
Evaluates on held-out test set with proper train/test split.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import math

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.getcwd())
from models.recommender import TwoTowerModel, InteractionDataset, RecipeDataset, collate_fn


def calculate_ranking_metrics(scores, target_idx, k_values=[5, 10, 20]):
    """
    Calculate Precision@K, Recall@K, NDCG@K for a single user.

    Args:
        scores: [num_items] tensor of scores
        target_idx: int, index of the ground truth item
        k_values: list of K values to evaluate

    Returns:
        dict of metrics for each K
    """
    metrics = {}

    # Get top-K indices
    _, top_k_indices = torch.topk(scores, max(k_values))
    top_k_indices = top_k_indices.cpu().numpy()

    for k in k_values:
        top_k = top_k_indices[:k]

        # Check if target is in top-K
        hit = 1 if target_idx in top_k else 0

        # Precision@K (binary relevance)
        metrics[f'precision@{k}'] = hit / k

        # Recall@K (always 1 or 0 for single target)
        metrics[f'recall@{k}'] = hit

        # NDCG@K
        if hit:
            rank = np.where(top_k == target_idx)[0][0]
            dcg = 1.0 / math.log2(rank + 2)  # +2 because rank is 0-indexed
            idcg = 1.0  # Ideal DCG for single relevant item
            metrics[f'ndcg@{k}'] = dcg / idcg
        else:
            metrics[f'ndcg@{k}'] = 0.0

    return metrics


def evaluate_full_ranking(model, test_loader, recipe_embeddings, device, num_samples=None):
    """
    Evaluate model using full-corpus ranking (not in-batch).
    This is the proper way to evaluate recommender systems.

    Args:
        model: Trained TwoTowerModel
        test_loader: DataLoader for test set
        recipe_embeddings: [num_recipes, embedding_dim] precomputed recipe embeddings
        device: torch device
        num_samples: Optional limit on number of test samples (for speed)

    Returns:
        dict of average metrics
    """
    model.eval()

    all_metrics = {
        'precision@5': [], 'precision@10': [], 'precision@20': [],
        'recall@5': [], 'recall@10': [], 'recall@20': [],
        'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
    }

    samples_evaluated = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_idx = batch['user_idx'].to(device)
            recipe_idx = batch['recipe_idx'].to(device)

            history_recipe_indices = batch['history_recipe_indices'].to(device)
            history_ingredients = batch['history_ingredients'].to(device)
            history_nutrition = batch['history_nutrition'].to(device)
            history_mask = batch['history_mask'].to(device)

            batch_size = user_idx.size(0)

            # Get user embeddings (with history)
            # We need to pass dummy recipe data for the forward pass, but we only need user embeddings
            ingredients = batch['ingredients'].to(device)
            nutrition = batch['nutrition'].to(device)

            user_emb, _ = model(
                user_idx, recipe_idx, ingredients, nutrition,
                history_recipe_indices, history_ingredients,
                history_nutrition, history_mask
            )

            # Compute scores against ALL recipes
            # scores: [batch_size, num_recipes]
            scores = torch.matmul(user_emb, recipe_embeddings.T)

            # For each user in batch
            for i in range(batch_size):
                target_idx = recipe_idx[i].item()
                user_scores = scores[i]

                # Calculate metrics
                metrics = calculate_ranking_metrics(user_scores, target_idx)

                for key, value in metrics.items():
                    all_metrics[key].append(value)

                samples_evaluated += 1
                if num_samples and samples_evaluated >= num_samples:
                    break

            if num_samples and samples_evaluated >= num_samples:
                break

    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    return avg_metrics, samples_evaluated


def precompute_recipe_embeddings(model, recipe_dataset, device, batch_size=1024):
    """
    Precompute embeddings for all recipes.

    Args:
        model: TwoTowerModel
        recipe_dataset: RecipeDataset
        device: torch device
        batch_size: batch size for processing

    Returns:
        [num_recipes, embedding_dim] tensor of recipe embeddings
    """
    model.eval()

    num_recipes = len(recipe_dataset)
    all_embeddings = []

    print("Precomputing recipe embeddings...")

    with torch.no_grad():
        for start_idx in tqdm(range(0, num_recipes, batch_size)):
            end_idx = min(start_idx + batch_size, num_recipes)

            # Collect batch
            batch_recipe_idx = []
            batch_ingredients = []
            batch_nutrition = []

            for idx in range(start_idx, end_idx):
                recipe_data = recipe_dataset[idx]
                batch_recipe_idx.append(recipe_data['recipe_idx'])
                batch_ingredients.append(recipe_data['ingredients'])
                batch_nutrition.append(recipe_data['nutrition'])

            # Stack tensors
            batch_recipe_idx = torch.stack(batch_recipe_idx).to(device)
            batch_nutrition = torch.stack(batch_nutrition).to(device)

            # Pad ingredients
            max_len = max([len(ing) for ing in batch_ingredients])
            padded_ingredients = torch.zeros((len(batch_ingredients), max_len), dtype=torch.long).to(device)
            for i, ing in enumerate(batch_ingredients):
                padded_ingredients[i, :len(ing)] = ing.to(device)

            # Get embeddings
            embeddings = model.get_recipe_embedding(batch_recipe_idx, padded_ingredients, batch_nutrition)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).to(device)


def main():
    print("="*80)
    print("TWO-TOWER MODEL EVALUATION (WITH HISTORY ENCODING)")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Paths
    DATA_DIR = "data/food.com-interaction"
    RECIPES_CSV = os.path.join(DATA_DIR, "RAW_recipes.csv")
    INTERACTIONS_CSV = os.path.join(DATA_DIR, "RAW_interactions.csv")
    MODEL_PATH = "models/saved/recommender_model_pytorch.pth"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_recommender_script.py")
        return

    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    embedding_dim = checkpoint['embedding_dim']

    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Trained epochs: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best validation NDCG@10: {checkpoint.get('best_ndcg', 'unknown'):.4f}")
    print()

    # Load datasets
    print("Loading datasets...")
    recipe_ds = RecipeDataset(RECIPES_CSV)

    test_ds = InteractionDataset(
        INTERACTIONS_CSV, INTERACTIONS_CSV, recipe_ds,
        split='test', train_ratio=0.8
    )

    print(f"\nTest set statistics:")
    print(f"  Test interactions: {len(test_ds)}")
    print(f"  Test users: {len(test_ds.unique_user_ids)}")
    print()

    # Create test loader
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Build model
    print("Building model...")
    model = TwoTowerModel(
        num_users=len(checkpoint['user_to_idx']),
        num_recipes=len(recipe_ds),
        vocab_size=len(checkpoint['vocab']),
        embedding_dim=embedding_dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Precompute recipe embeddings
    recipe_embeddings = precompute_recipe_embeddings(model, recipe_ds, device)
    print(f"  Recipe embeddings shape: {recipe_embeddings.shape}")
    print()

    # Evaluate on test set
    print("="*80)
    print("FULL RANKING EVALUATION ON TEST SET")
    print("="*80)
    print()

    # Evaluate on subset for speed (adjust num_samples for full evaluation)
    metrics, num_samples = evaluate_full_ranking(
        model, test_loader, recipe_embeddings, device,
        num_samples=5000  # Set to None for full test set
    )

    # Print results
    print(f"\nEvaluation Results ({num_samples} samples):")
    print("="*80)

    print("\nPrecision @ K:")
    for k in [5, 10, 20]:
        print(f"  Precision@{k:2d}: {metrics[f'precision@{k}']:.6f}")

    print("\nRecall @ K:")
    for k in [5, 10, 20]:
        print(f"  Recall@{k:2d}:    {metrics[f'recall@{k}']:.6f}")

    print("\nNDCG @ K:")
    for k in [5, 10, 20]:
        print(f"  NDCG@{k:2d}:      {metrics[f'ndcg@{k}']:.6f}")

    print("\n" + "="*80)

    # Compare with previous broken model
    print("\nComparison with broken model:")
    print(f"  Previous NDCG@10: 0.00006 (essentially random)")
    print(f"  Current NDCG@10:  {metrics['ndcg@10']:.6f}")

    if metrics['ndcg@10'] > 0.001:
        improvement = metrics['ndcg@10'] / 0.00006
        print(f"  Improvement:      {improvement:.1f}x better!")

    print("\n" + "="*80)

    # Hit rate analysis
    hit_rate_5 = metrics['recall@5']
    hit_rate_10 = metrics['recall@10']
    hit_rate_20 = metrics['recall@20']

    print("\nHit Rate (% of users with relevant item in top-K):")
    print(f"  Top-5:  {hit_rate_5*100:.2f}%")
    print(f"  Top-10: {hit_rate_10*100:.2f}%")
    print(f"  Top-20: {hit_rate_20*100:.2f}%")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
