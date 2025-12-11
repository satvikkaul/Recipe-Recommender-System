"""
Test script to verify the updated User Tower with history encoding works properly.
This script tests:
1. Model can be instantiated
2. Forward pass with history works
3. Forward pass without history works (cold-start scenario)
4. Gradients flow properly
5. Training loop runs without errors
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.getcwd())
from models.recommender import TwoTowerModel, InteractionDataset, RecipeDataset, collate_fn

def test_model_architecture():
    """Test 1: Model instantiation"""
    print("\n" + "="*60)
    print("TEST 1: Model Instantiation")
    print("="*60)

    try:
        model = TwoTowerModel(
            num_users=100,
            num_recipes=1000,
            vocab_size=500,
            embedding_dim=32
        )
        print("✓ Model created successfully")
        print(f"  - User Tower: {sum(p.numel() for p in model.user_tower.parameters())} parameters")
        print(f"  - Recipe Tower: {sum(p.numel() for p in model.recipe_tower.parameters())} parameters")
        print(f"  - Total: {sum(p.numel() for p in model.parameters())} parameters")
        return True
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False

def test_forward_with_history():
    """Test 2: Forward pass with history"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass WITH History")
    print("="*60)

    try:
        model = TwoTowerModel(
            num_users=100,
            num_recipes=1000,
            vocab_size=500,
            embedding_dim=32
        )

        batch_size = 16
        history_len = 10
        max_ing_len = 20

        # Create dummy inputs
        user_indices = torch.randint(0, 100, (batch_size,))
        recipe_indices = torch.randint(0, 1000, (batch_size,))
        ingredients = torch.randint(0, 500, (batch_size, max_ing_len))
        nutrition = torch.randn(batch_size, 7)

        # History
        history_recipe_indices = torch.randint(0, 1000, (batch_size, history_len))
        history_ingredients = torch.randint(0, 500, (batch_size, history_len, max_ing_len))
        history_nutrition = torch.randn(batch_size, history_len, 7)
        history_mask = torch.ones(batch_size, history_len, dtype=torch.bool)

        # Forward pass
        user_emb, recipe_emb = model(
            user_indices, recipe_indices, ingredients, nutrition,
            history_recipe_indices, history_ingredients, history_nutrition, history_mask
        )

        print(f"✓ Forward pass successful")
        print(f"  - User embeddings shape: {user_emb.shape}")
        print(f"  - Recipe embeddings shape: {recipe_emb.shape}")
        print(f"  - Expected: torch.Size([{batch_size}, 32])")

        assert user_emb.shape == (batch_size, 32), f"User embedding shape mismatch"
        assert recipe_emb.shape == (batch_size, 32), f"Recipe embedding shape mismatch"
        print("✓ Output shapes correct")

        return True
    except Exception as e:
        print(f"✗ Forward pass with history failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_without_history():
    """Test 3: Forward pass without history (cold-start)"""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass WITHOUT History (Cold-Start)")
    print("="*60)

    try:
        model = TwoTowerModel(
            num_users=100,
            num_recipes=1000,
            vocab_size=500,
            embedding_dim=32
        )

        batch_size = 16
        max_ing_len = 20

        # Create dummy inputs (no history)
        user_indices = torch.randint(0, 100, (batch_size,))
        recipe_indices = torch.randint(0, 1000, (batch_size,))
        ingredients = torch.randint(0, 500, (batch_size, max_ing_len))
        nutrition = torch.randn(batch_size, 7)

        # Forward pass without history
        user_emb, recipe_emb = model(
            user_indices, recipe_indices, ingredients, nutrition
        )

        print(f"✓ Forward pass successful (cold-start mode)")
        print(f"  - User embeddings shape: {user_emb.shape}")
        print(f"  - Recipe embeddings shape: {recipe_emb.shape}")

        assert user_emb.shape == (batch_size, 32), f"User embedding shape mismatch"
        assert recipe_emb.shape == (batch_size, 32), f"Recipe embedding shape mismatch"
        print("✓ Cold-start mode works correctly")

        return True
    except Exception as e:
        print(f"✗ Forward pass without history failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test 4: Gradient flow"""
    print("\n" + "="*60)
    print("TEST 4: Gradient Flow")
    print("="*60)

    try:
        model = TwoTowerModel(
            num_users=100,
            num_recipes=1000,
            vocab_size=500,
            embedding_dim=32
        )

        batch_size = 16
        history_len = 10
        max_ing_len = 20

        # Create dummy inputs
        user_indices = torch.randint(0, 100, (batch_size,))
        recipe_indices = torch.randint(0, 1000, (batch_size,))
        ingredients = torch.randint(0, 500, (batch_size, max_ing_len))
        nutrition = torch.randn(batch_size, 7)

        history_recipe_indices = torch.randint(0, 1000, (batch_size, history_len))
        history_ingredients = torch.randint(0, 500, (batch_size, history_len, max_ing_len))
        history_nutrition = torch.randn(batch_size, history_len, 7)
        history_mask = torch.ones(batch_size, history_len, dtype=torch.bool)

        # Forward pass
        user_emb, recipe_emb = model(
            user_indices, recipe_indices, ingredients, nutrition,
            history_recipe_indices, history_ingredients, history_nutrition, history_mask
        )

        # Compute loss (in-batch negative sampling)
        scores = torch.matmul(user_emb, recipe_emb.T)
        labels = torch.arange(batch_size)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores, labels)

        # Backward pass
        loss.backward()

        print(f"✓ Backward pass successful")
        print(f"  - Loss: {loss.item():.4f}")

        # Check gradients exist
        user_tower_has_grad = any(p.grad is not None for p in model.user_tower.parameters())
        recipe_tower_has_grad = any(p.grad is not None for p in model.recipe_tower.parameters())

        print(f"  - User tower has gradients: {user_tower_has_grad}")
        print(f"  - Recipe tower has gradients: {recipe_tower_has_grad}")

        assert user_tower_has_grad, "User tower has no gradients!"
        assert recipe_tower_has_grad, "Recipe tower has no gradients!"
        print("✓ Gradients flowing properly")

        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_loop_subset():
    """Test 5: Training loop on small data subset"""
    print("\n" + "="*60)
    print("TEST 5: Training Loop on Small Subset")
    print("="*60)

    try:
        DATA_DIR = "data/food.com-interaction"
        RECIPES_CSV = os.path.join(DATA_DIR, "RAW_recipes.csv")
        INTERACTIONS_CSV = os.path.join(DATA_DIR, "RAW_interactions.csv")

        if not os.path.exists(RECIPES_CSV) or not os.path.exists(INTERACTIONS_CSV):
            print("⊘ Skipping test (data files not found)")
            return True

        print("Loading small data subset...")
        recipe_ds = RecipeDataset(RECIPES_CSV)
        interaction_ds = InteractionDataset(INTERACTIONS_CSV, INTERACTIONS_CSV, recipe_ds, max_history=5)

        # Use small batch for testing
        BATCH_SIZE = 64
        train_loader = DataLoader(
            interaction_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        print(f"  - Users: {len(interaction_ds.unique_user_ids)}")
        print(f"  - Recipes: {len(recipe_ds)}")
        print(f"  - Interactions: {len(interaction_ds)}")

        # Create model
        model = TwoTowerModel(
            num_users=len(interaction_ds.unique_user_ids),
            num_recipes=len(recipe_ds),
            vocab_size=len(recipe_ds.vocab),
            embedding_dim=32
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train for 1 batch only
        print("\nRunning 1 training iteration...")
        model.train()

        batch = next(iter(train_loader))

        user_idx = batch['user_idx']
        recipe_idx = batch['recipe_idx']
        ingredients = batch['ingredients']
        nutrition = batch['nutrition']
        history_recipe_indices = batch['history_recipe_indices']
        history_ingredients = batch['history_ingredients']
        history_nutrition = batch['history_nutrition']
        history_mask = batch['history_mask']

        optimizer.zero_grad()

        user_emb, recipe_emb = model(
            user_idx, recipe_idx, ingredients, nutrition,
            history_recipe_indices, history_ingredients,
            history_nutrition, history_mask
        )

        scores = torch.matmul(user_emb, recipe_emb.T)
        labels = torch.arange(scores.size(0))

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        print(f"✓ Training iteration successful")
        print(f"  - Batch size: {user_idx.size(0)}")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - History used: {history_mask.sum().item()} items")

        return True

    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# USER TOWER WITH HISTORY ENCODING - TEST SUITE")
    print("#"*60)

    tests = [
        test_model_architecture,
        test_forward_with_history,
        test_forward_without_history,
        test_gradient_flow,
        test_training_loop_subset
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The user tower with history encoding is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
