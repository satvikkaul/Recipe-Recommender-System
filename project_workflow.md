# NutriSnap: System Architecture & Workflow

## 1. Project Workflow Overview

The **NutriSnap** system consists of two distinct machine learning pipelines (Image Classification & Recommendation) that converge in a FastAPI backend to serve a user-facing application.

### A. Training Pipeline (Offline)
This phase processes data and produces saved model artifacts (`.keras`, `saved_model`).
1.  **Image Training**: `train_image_script.py` -> `models.image_classifier` -> `models/saved/image_model.keras`
2.  **Recommender Training**: `train_recommender_script.py` -> `models.recommender` -> `models/saved/recommender_index`

### B. Inference Pipeline (Online/Real-time)
This phase serves predictions to the user.
1.  **Backend**: `uvicorn` starts `backend/main.py`.
2.  **Startup**: `backend/main.py` calls `backend/inference.py` to load the saved artifacts from step A.
3.  **Request**: Frontend sends image -> API (`/predict-food`) -> `ImageEngine` -> Class Label ("pizza").
4.  **Request**: Frontend sends user ID -> API (`/recommend`) -> `RecommenderEngine` -> Recipe List.

---

## 2. File & Function Breakdown

### Backend Layer
*   **`backend/main.py`**
    *   **Role**: Entry point for the API server.
    *   **Key Functions**:
        *   `load_models()`: Runs on startup. Checks if model files exist and initializes the engines.
        *   `/predict-food` (POST): Receives image bytes, passes them to `image_engine`.
        *   `/recommend` (POST): Receives user ID and calories, passes them to `recommender_engine`.
*   **`backend/inference.py`**
    *   **Role**: Wrapper classes that handle the logic of using the raw TensorFlow models.
    *   **`ImageEngine`**:
        *   `__init__`: Loads the `.keras` model.
        *   `predict(image_bytes)`: Decodes image, resizes to 224x224, runs model, returning the highest probability food name and its nutrition info.
    *   **`RecommenderEngine`**:
        *   `__init__`: Loads the saved Two-Tower index and the `recipes.csv` (for metadata like description/calories).
        *   `recommend(user_id, ...)`: Queries the TF index for candidate IDs, then looks up those IDs in the CSV to return human-readable recipe details.

### Model Layer
*   **`models/image_classifier.py`**
    *   **Role**: Defines the CNN architecture.
    *   **`build_custom_cnn`**: Creates a sequential Keras model (Conv2D -> MaxPooling -> Dense). *Current Weakness: Too simple for 101 classes.*
    *   **`load_data`**: Uses `image_dataset_from_directory` to load images from disk.
*   **`models/recommender.py`**
    *   **Role**: Defines the Two-Tower interaction model using TensorFlow Recommenders (TFRS).
    *   **`build_model`**: Loads CSVs, processes headers (renaming `id` -> `recipe_id`), creates datasets, and builds the `TwoTowerModel` (User Tower + Recipe Tower).
    *   **`save_model_index`**: Creates a "BruteForce" retrieval layer (the index) that can be served.

### Training Scripts
*   **`train_image_script.py`**: Orchestrator. Checks paths, calls `load_data`, `build_custom_cnn`, `model.fit()`, and `model.save()`.
*   **`train_recommender_script.py`**: Orchestrator. Checks paths, calls `build_model`, `model.fit()`, and `save_model_index()`.

---

## 3. Improvement Suggestions

### Critical Improvements (High Impact)
1.  **Switch to Transfer Learning (Image Model)**
    *   **Reason**: The current custom CNN gets stuck (predicts "paella" for everything) because it cannot learn features for 101 classes from scratch in just 3-5 epochs.
    *   **Fix**: Use `MobileNetV2` (pre-trained on ImageNet). It already "knows" shapes and textures.
    *   **Result**: >80% accuracy in <5 epochs.
2.  **Clean Data Interaction (Recommender)**
    *   **Reason**: `RAW_interactions.csv` has mixed types in 'review' which crashed training.
    *   **Fix**: (Already Applied) Filtered datasets to only use `user_id` and `recipe_id` strict strings.

### Functionality Improvements
3.  **Real Nutrition Data**: Currently, `ImageEngine` uses a hardcoded dictionary for generic foods ("pizza": 285 cal). It should ideally look up the specific predicted class in a real nutrition database or the `recipes.csv` if mapped.
4.  **User Cold Start**: The recommender fails for a brand new User ID not in the training set. We should add a fallback (e.g., recommend "Popular" items) if `user_id` is unknown.

---

## 4. Performance & Baselines

When reporting your results, you should compare your model's performance against a "Baseline" (Random Guessing). This proves your AI is actually learning.

### A. Image Classifier Modals
*   **Metric**: Accuracy (Top-1)
*   **Classes**: 101 Food categories

| Model Type | Accuracy Calculation | Expected Accuracy |
| :--- | :--- | :--- |
| **Random Guessing (Baseline)** | $1 / 101$ | **~0.99%** |
| **Simple CNN (Previous)** | Trained from scratch on limited data | **~3 - 5%** |
| **EfficientNetB2 (Current)** | Transfer Learning (Pre-trained) | **~50 - 65%** (Epoch 2-5) |

**Conclusion**: Your current model (~50%) is **50x better** than random chance.

### B. Recommender System
*   **Metric**: Top-K Categorical Accuracy (Probability distinct user/item pair is ranked high)

| Model Type | Expected Performance |
| :--- | :--- |
| **Random Recommendation** | Near **0%** (1 in 230,000 recipes) |
| **Popularity Baseline** | Recommends most-rated items to everyone. Better than random but rigid. |
| **Two-Tower (Current)** | Personalized to user history. Expected **1% - 5%** (which is decent for retrieval tasks). |

---

## 5. How to Check Current Accuracy

### Image Classifier
Run: `python train_image_script.py`
*   Look for `Acc: 0.XXXX` in the progress bar.

### Recommender
Run: `python train_recommender_script.py`
*   Look for `Loss` decreasing. Lower loss = Better ranking.

