Whats upppppppp bro

# NutriSnap - AI Nutrition Assistant

NutriSnap is an intelligent nutrition assistant that combines computer vision for food recognition with a personalized recipe recommender system.

## 🚀 Features
*   **Predict Food**: Identify food items from images (e.g., pizza, sushi, salad) using a custom MobileNetV2 classifier.
*   **Recommend Recipes**: Get personalized recipe suggestions based on your remaining calorie budget using a Two-Tower Neural Network.

## 🛠 Tech Stack
*   **Backend**: FastAPI
*   **ML Framework**: TensorFlow / Keras / TensorFlow Recommenders (TFRS)
*   **Data**: Food-101 (Images) & Food.com (Recommender Interactions)

## 📂 Project Structure
*   `backend/`: FastAPI application and inference engines.
*   `models/`: TensorFlow model definitions.
*   `data/`: Data storage (Git-ignored large datasets).
*   `train_image_script.py`: Training script for the image classifier.
*   `train_recommender_script.py`: Training script for the recommender system.

## 🏃‍♂️ Quick Start

1.  **Install Dependencies**
    ```bash
    pip install tensorflow tensorflow-recommenders fastapi uvicorn pandas numpy matplotlib
    ```

2.  **Run Backend**
    ```bash
    uvicorn backend.main:app --reload
    ```

3.  **API Docs**
    *   Open `http://localhost:8000/docs` to test endpoints.
