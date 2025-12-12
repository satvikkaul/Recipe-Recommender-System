"""
Create synthetic demo users with specific cuisine preferences.

This script generates 3 demo users:
- demo_user_italian: Prefers Italian cuisine (pasta, pizza, risotto, etc.)
- demo_user_indian: Prefers Indian cuisine (curry, biryani, tikka, etc.)
- demo_user_american: Prefers American cuisine (burger, BBQ, mac and cheese, etc.)

Each user gets 20 realistic interactions with high ratings (4-5 stars).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration
RECIPES_CSV = "data/food.com-interaction/RAW_recipes.csv"
INTERACTIONS_CSV = "data/food.com-interaction/RAW_interactions.csv"
OUTPUT_CSV = "data/food.com-interaction/demo_users_interactions.csv"

# Demo users
DEMO_USERS = {
    "demo_user_italian": {
        "keywords": ["italian", "pasta", "pizza", "risotto", "lasagna", "spaghetti",
                    "ravioli", "pesto", "marinara", "parmesan", "mozzarella",
                    "alfredo", "carbonara", "bolognese", "gnocchi"],
        "interactions": 20
    },
    "demo_user_indian": {
        "keywords": ["indian", "curry", "tikka", "masala", "biryani", "tandoori",
                    "naan", "samosa", "dal", "paneer", "korma", "vindaloo",
                    "chai", "chutney", "turmeric", "cumin", "garam masala"],
        "interactions": 20
    },
    "demo_user_american": {
        "keywords": ["burger", "bbq", "barbecue", "mac and cheese", "fried chicken",
                    "meatloaf", "pot roast", "chili", "corn bread", "apple pie",
                    "brownie", "pulled pork", "ribs", "coleslaw", "biscuit"],
        "interactions": 20
    }
}

def load_recipes():
    """Load recipes dataset"""
    print(f"Loading recipes from {RECIPES_CSV}...")
    recipes = pd.read_csv(RECIPES_CSV)
    print(f"Loaded {len(recipes)} recipes")
    return recipes

def find_recipes_by_cuisine(recipes, keywords, n=20):
    """Find recipes matching cuisine keywords"""
    # Create a combined search pattern (case-insensitive)
    pattern = '|'.join(keywords)

    # Search in recipe names and tags
    matches = recipes[
        recipes['name'].str.lower().str.contains(pattern, na=False, case=False) |
        recipes['tags'].str.lower().str.contains(pattern, na=False, case=False)
    ]

    print(f"  Found {len(matches)} matching recipes")

    # Sample n recipes if we have enough, otherwise return all
    if len(matches) >= n:
        return matches.sample(n=n, random_state=42)
    else:
        print(f"  Warning: Only {len(matches)} matches found, requested {n}")
        return matches

def generate_interactions(user_id, recipes, n_interactions=20):
    """Generate realistic interactions for a user"""
    interactions = []

    # Generate interactions over the past 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    for i, (_, recipe) in enumerate(recipes.iterrows()):
        if i >= n_interactions:
            break

        # Generate a random date within the range
        random_days = random.randint(0, 180)
        interaction_date = start_date + timedelta(days=random_days)

        # High ratings for preferred cuisine (4-5 stars)
        rating = random.choice([4, 4, 4, 5, 5])  # Weighted towards 4-5

        # Optional review (60% chance)
        review = ""
        if random.random() < 0.6:
            reviews_pool = [
                "Delicious! Will make again.",
                "Great recipe, family loved it!",
                "Perfect, followed exactly as written.",
                "Amazing flavors!",
                "Best recipe I've tried!",
                "Absolutely loved this dish.",
                "Easy to make and tasty.",
                "Wonderful, highly recommend!",
                "Fantastic, made it twice already.",
                "Excellent, will be making this regularly."
            ]
            review = random.choice(reviews_pool)

        interactions.append({
            "user_id": user_id,
            "recipe_id": recipe['id'],
            "date": interaction_date.strftime("%Y-%m-%d"),
            "rating": rating,
            "review": review
        })

    return interactions

def main():
    # Load recipes
    recipes = load_recipes()

    all_interactions = []

    # Generate interactions for each demo user
    for user_id, config in DEMO_USERS.items():
        print(f"\nGenerating interactions for {user_id}...")
        print(f"  Cuisine keywords: {', '.join(config['keywords'][:5])}...")

        # Find matching recipes
        matching_recipes = find_recipes_by_cuisine(
            recipes,
            config['keywords'],
            n=config['interactions']
        )

        # Generate interactions
        user_interactions = generate_interactions(
            user_id,
            matching_recipes,
            n_interactions=config['interactions']
        )

        all_interactions.extend(user_interactions)
        print(f"  Generated {len(user_interactions)} interactions")

    # Create DataFrame
    interactions_df = pd.DataFrame(all_interactions)

    # Sort by date
    interactions_df = interactions_df.sort_values('date')

    # Save to CSV
    print(f"\nSaving {len(interactions_df)} interactions to {OUTPUT_CSV}...")
    interactions_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Demo user interactions saved successfully!")

    # Print summary
    print("\n" + "="*60)
    print("DEMO USERS CREATED:")
    print("="*60)
    for user_id in DEMO_USERS.keys():
        user_data = interactions_df[interactions_df['user_id'] == user_id]
        avg_rating = user_data['rating'].mean()
        print(f"\n{user_id}:")
        print(f"  Total interactions: {len(user_data)}")
        print(f"  Average rating: {avg_rating:.1f}/5")
        print(f"  Date range: {user_data['date'].min()} to {user_data['date'].max()}")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Append demo interactions to main interactions file:")
    print(f"   cat {OUTPUT_CSV} >> {INTERACTIONS_CSV}")
    print("\n2. Retrain the recommender model to include demo users")
    print("\n3. Or manually add these users to the model's user_to_idx mapping")
    print("="*60)

if __name__ == "__main__":
    main()
