import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
# Import new components (collate_fn)
from models.recommender import TwoTowerModel, InteractionDataset, RecipeDataset, collate_fn

# -------------------------------------------------------------------
# Metrics Implementation
# -------------------------------------------------------------------
def calculate_metrics(k, scores, labels):
    """
    Calculates Precision@K, Recall@K, NDCG@K for a batch.
    
    Args:
        scores: [Batch, Batch] matrix (User vs All Items in Batch)
        labels: [Batch] vector of correct indices (diagonal)
    
    Note: In-batch evaluation is an approximation. 
    It checks if the correct item is ranked higher than other random items in the batch.
    """
    batch_size = scores.size(0)
    
    # Get Top K indices
    # values, indices: [Batch, K]
    _, top_k_indices = torch.topk(scores, k, dim=1)
    
    # Expand labels to check against top K
    # labels: [Batch, 1]
    targets = labels.view(-1, 1).expand_as(top_k_indices)
    
    # Check hits
    hits = (top_k_indices == targets).float() # [Batch, K]
    
    # 1. Precision@K: (Hits / K)
    # Since there is only 1 correct item per user in this batch setting:
    # If hit, precision is 1/K. If miss, 0.
    # This definition varies. Often Precision is 'Relevant Items / K'.
    precision_k = hits.sum(dim=1) / k
    
    # 2. Recall@K: (Hits / Total Relevant)
    # Total Relevant is always 1 here.
    recall_k = hits.sum(dim=1) / 1.0 
    
    # 3. NDCG@K
    # IDCG is always 1.0 because optimal ranking puts the 1 relevant item at pos 0.
    # DCG = rel_i / log2(i+2)
    # rel_i is 1 if hit, 0 otherwise.
    # We need the position of the hit.
    
    # Create rank weights: 1/log2(2), 1/log2(3)...
    weights = 1.0 / torch.log2(torch.arange(2, k + 2).float().to(scores.device))
    dcg = (hits * weights).sum(dim=1)
    idcg = 1.0
    ndcg_k = dcg / idcg
    
    return precision_k.mean().item(), recall_k.mean().item(), ndcg_k.mean().item()

def train_one_epoch(model, loader, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    metrics = {"p10": 0, "r10": 0, "n10": 0}
    count = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    criterion = nn.CrossEntropyLoss()
    
    for batch in loop:
        # Unpack Data
        user_idx = batch['user_idx'].to(device)
        recipe_idx = batch['recipe_idx'].to(device)
        ingredients = batch['ingredients'].to(device)
        nutrition = batch['nutrition'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        user_emb, recipe_emb = model(user_idx, recipe_idx, ingredients, nutrition)
        
        # Scores
        scores = torch.matmul(user_emb, recipe_emb.T)
        labels = torch.arange(scores.size(0)).to(device)
        
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Metrics (approximate on batch)
        if batch['user_idx'].size(0) > 10: # Only calc if batch big enough
            p, r, n = calculate_metrics(10, scores, labels)
            metrics["p10"] += p
            metrics["r10"] += r
            metrics["n10"] += n
            count += 1
        
        loop.set_postfix(loss=loss.item(), ndcg=metrics["n10"]/(count+1e-6))
        
    avg_metrics = {k: v/count for k,v in metrics.items()}
    return running_loss / len(loader), avg_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BATCH_SIZE = 4096 # slightly smaller for complex model memory
    EPOCHS = 5
    EMBEDDING_DIM = 32
    
    DATA_DIR = "data/food.com-interaction"
    RECIPES_CSV = os.path.join(DATA_DIR, "RAW_recipes.csv")
    INTERACTIONS_CSV = os.path.join(DATA_DIR, "RAW_interactions.csv")
    MODEL_SAVE_PATH = "models/saved/recommender_model_pytorch.pth"
    
    if not os.path.exists(RECIPES_CSV):
        print("Error: Data not found.")
        return
        
    # 1. Initialize Datasets (Builds Vocab too)
    print("Loading Maps & Processing features (this may take a moment)...")
    recipe_ds = RecipeDataset(RECIPES_CSV)
    interaction_ds = InteractionDataset(INTERACTIONS_CSV, INTERACTIONS_CSV, recipe_ds)
    
    # 2. Loader with Collate
    train_loader = DataLoader(
        interaction_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    
    print(f"\nData Loaded:")
    print(f"Users: {len(interaction_ds.unique_user_ids)}")
    print(f"Recipes: {len(recipe_ds)}")
    print(f"Ingredient Vocab: {len(recipe_ds.vocab)}")
    
    # 3. Model
    model = TwoTowerModel(
        num_users=len(interaction_ds.unique_user_ids),
        num_recipes=len(recipe_ds),
        vocab_size=len(recipe_ds.vocab),
        embedding_dim=EMBEDDING_DIM
    )
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam is generally better for mixed types
    
    # 4. Train
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        loss, mets = train_one_epoch(model, train_loader, optimizer, device, epoch, EPOCHS)
        print(f"Result | Loss: {loss:.4f} | NDCG@10: {mets['n10']:.4f} | Recall@10: {mets['r10']:.4f}")
        
    # 5. Save (Must save Vocab too!)
    print("Saving Model...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'user_to_idx': interaction_ds.user_to_idx,
        'recipe_to_idx': recipe_ds.recipe_to_idx,
        'vocab': recipe_ds.vocab,
        'embedding_dim': EMBEDDING_DIM
    }
    torch.save(save_dict, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
if __name__ == "__main__":
    main()
