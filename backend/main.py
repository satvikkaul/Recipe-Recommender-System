import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

class RecommendationRequest(BaseModel):
    user_id: str
    current_calories: int
    daily_goal: int = 2000

@app.post("/recommend")
def get_recommendations(req: RecommendationRequest):
    if not recommender_engine:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")
    
    try:
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
