# Data Instructions

## Recommender System Data (Food.com)

To train the Recommender System with real data:

1.  Download the **Food.com Recipes and Interactions** dataset (e.g., from Kaggle).
    *   Files needed: `RAW_recipes.csv` and `RAW_interactions.csv`.
2.  Place them in the `data/` directory.
3.  Update the data loading logic in `backend/` and `notebooks/` to point to these files instead of `data/mock/`.

## Image Recognition Data (Food-101)

To train the Image Model with real data:

1.  Download the **Food-101** dataset.
2.  Extract it. The structure should be a folder containing 101 subfolders (one for each food category).
3.  Place the extracted `images` folder at `data/food-101/images`.
4.  Update `models/image_classifier.py` (or the training notebook) to point to `data/food-101/images`.
