import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil

# Import engines
from backend.inference import RecommenderEngine, ImageEngine

app = FastAPI(title="NutriSnap API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
# Paths need to be correct relative to where we run the server
RECOMMENDER_PATH = "models/saved/recommender_model_pytorch.pth"
RECIPES_CSV = "data/food.com-interaction/RAW_recipes.csv"
INTERACTIONS_CSV = "data/food.com-interaction/RAW_interactions.csv"
IMAGE_MODEL_PATH = "models/saved/image_model_pytorch.pth"

# Dynamically load class names from the dataset directory if available
# Otherwise fall back to a default list (or load from a saved metadata file)
DATA_DIR_FOOD101 = "data/food-101/images"
if os.path.exists(DATA_DIR_FOOD101):
    CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR_FOOD101) if os.path.isdir(os.path.join(DATA_DIR_FOOD101, d))])
else:
    # Fallback or Mock
    CLASS_NAMES = ['burger', 'pasta', 'pizza', 'salad', 'sushi']

# Lazy loading or startup event
recommender_engine = None
image_engine = None

@app.on_event("startup")
def load_models():
    global recommender_engine, image_engine
    if os.path.exists(RECOMMENDER_PATH):
        recommender_engine = RecommenderEngine(
            RECOMMENDER_PATH,
            RECIPES_CSV,
            interactions_csv=INTERACTIONS_CSV
        )
        # Load demo users automatically
        demo_file = "data/food.com-interaction/demo_users_interactions.csv"
        if os.path.exists(demo_file):
            recommender_engine.load_demo_users(demo_file)
        else:
            print(f"Info: No demo users file found at {demo_file}")
    else:
        print(f"Warning: {RECOMMENDER_PATH} not found.")

    # Try both .keras and SavedModel formats
    image_path = IMAGE_MODEL_PATH
    if os.path.exists(image_path + ".keras"):
        image_path = image_path + ".keras"

    if os.path.exists(image_path):
        image_engine = ImageEngine(image_path, CLASS_NAMES, recipes_csv=RECIPES_CSV)
    else:
        print(f"Warning: Image model not found at {image_path}")

@app.get("/")
def read_root():
    return {"message": "Welcome to NutriSnap API"}

@app.get("/users/info")
def get_user_info():
    """Get information about available users in the recommender system"""
    if not recommender_engine:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")

    info = recommender_engine.get_user_info()
    return info

@app.get("/users/check/{user_id}")
def check_user_exists(user_id: str):
    """Check if a specific user ID exists in the recommender system"""
    if not recommender_engine:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")

    exists = recommender_engine.user_exists(user_id)
    history_count = len(recommender_engine.user_history.get(str(user_id), []))
    is_demo = hasattr(recommender_engine, 'demo_users') and str(user_id) in recommender_engine.demo_users

    return {
        "user_id": user_id,
        "exists": exists,
        "interactions": history_count if exists else 0,
        "is_demo": is_demo,
        "message": f"User '{user_id}' {'exists' if exists else 'does not exist'} in the system"
    }

@app.get("/users/demo")
def get_demo_users():
    """Get list of available demo users"""
    if not recommender_engine:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")

    demo_users = recommender_engine.get_demo_users()
    return {
        "demo_users": demo_users,
        "count": len(demo_users)
    }

class RecommendationRequest(BaseModel):
    user_id: str
    current_calories: int
    daily_goal: int = 2000
    food_name: Optional[str] = None  # Optional: food name from image classification

@app.post("/recommend")
def get_recommendations(req: RecommendationRequest):
    if not recommender_engine:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")

    try:
        # Use context-aware recommendations if food_name is provided
        if req.food_name:
            print(f"Using context-aware recommendations for food: {req.food_name}")
            results = recommender_engine.recommend_with_context(
                req.user_id,
                req.food_name,
                req.current_calories,
                req.daily_goal
            )
        else:
            # Fallback to regular recommendations (backward compatible)
            print("Using regular recommendations (no food context)")
            results = recommender_engine.recommend(
                req.user_id,
                req.current_calories,
                req.daily_goal
            )
        return {"recommendations": results}
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-food")
async def predict_food(file: UploadFile = File(...)):
    if not image_engine:
        raise HTTPException(status_code=503, detail="Image model not loaded")
    
    try:
        contents = await file.read()
        result = image_engine.predict(contents)
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
