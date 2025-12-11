# 🥗 NutriSnap - AI Nutrition Assistant

NutriSnap is an intelligent nutrition assistant that combines computer vision for food recognition with a personalized Two-Tower GRU recommender system. Built with PyTorch and FastAPI, it provides goal-aware recipe recommendations based on user history and caloric goals.

## ✨ Features

### 🍽️ **Image-Based Food Recognition**
- **EfficientNetB2** transfer learning model trained on Food-101 dataset
- Real-time food classification with confidence scores
- Instant nutritional information (calories, protein, carbs, fat)
- **Accuracy**: ~50-65% (50x better than random guessing)

### 🎯 **Personalized Recipe Recommendations**
- **Two-Tower Neural Architecture** with GRU history encoder
- Learns from 1.1M+ user interactions (Food.com dataset)
- **Goal-Aware Re-ranking**: Prioritizes recipes that fit your calorie budget
- **Content Features**: Ingredients (Top 2000 vocab) + 7 nutritional attributes
- **Performance**: 3.6x better NDCG@10 than popularity baseline

### 💎 **Modern Web Interface**
- Beautiful gradient-based UI with smooth animations
- Circular calorie progress tracker with SVG animations
- Real-time feedback and loading states
- Fully responsive (desktop, tablet, mobile)
- Drag-and-drop image upload

## 🛠 Tech Stack

### **Backend**
- **Framework**: FastAPI (Python 3.8+)
- **ML Framework**: PyTorch 2.0+ / TorchVision
- **Optimization**: Mixed Precision Training (AMP) for RTX 3060

### **Machine Learning**
- **Image Model**: EfficientNetB2 (pretrained on ImageNet)
- **Recommender**: Two-Tower with GRU (64D embeddings)
- **Training**: 15 epochs, batch=512, AdamW optimizer
- **Data**: Food-101 (101 classes) + Food.com (1.1M interactions)

### **Frontend**
- **Framework**: React 18 (via CDN)
- **Styling**: Modern CSS3 with gradients and animations
- **Design**: Card-based layout, responsive grid

## 📂 Project Structure

```
Recipe-Recommender-System/
├── backend/
│   ├── main.py              # FastAPI application
│   └── inference.py         # Inference engines (Image + Recommender)
├── models/
│   ├── image_classifier.py  # EfficientNetB2 definition
│   ├── recommender.py       # Two-Tower GRU model
│   └── saved/               # Trained model weights (git-ignored)
├── frontend/
│   ├── index.html           # Modern React UI
│   └── package.json         # Frontend dependencies
├── data/                    # Datasets (git-ignored)
│   ├── food-101/            # Image classification data
│   └── food.com-interaction/ # Recommendation data
├── train_image_script.py         # Train image classifier
├── train_recommender_script.py   # Train recommender
├── evaluate_baselines.py         # Baseline evaluation
├── evaluate_twotower.py          # Two-Tower evaluation
└── notebooks/                     # Jupyter notebooks
```

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
# PyTorch (with CUDA support for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Other dependencies
pip install fastapi uvicorn pandas numpy pillow scikit-learn tqdm
```

### 2. Download Datasets
See `DATA_INSTRUCTIONS.md` for dataset download and setup instructions.

### 3. Train Models (Optional - pretrained models available)
```bash
# Train image classifier (5 epochs, ~2 hours on RTX 3060)
python train_image_script.py

# Train recommender (15 epochs, ~2.5 hours on RTX 3060)
python train_recommender_script.py
```

### 4. Start Backend Server
```bash
uvicorn backend.main:app --reload

# Backend runs on: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### 5. Launch Frontend UI
```bash
# Option 1: Using launch script
./start_ui.sh

# Option 2: Manual Python server
cd frontend
python3 -m http.server 3000

# Open browser to: http://localhost:3000
```

**📖 For detailed UI guide, see:** `QUICK_START_UI.md`

## 🎯 Model Performance

### Image Classifier (EfficientNetB2)
- **Test Accuracy**: ~50-65%
- **Baseline (Random)**: 0.99% (1/101 classes)
- **Improvement**: **50x better** than random

### Two-Tower Recommender
| Metric | Popularity | KNN | Two-Tower (Ours) | Improvement |
|--------|-----------|-----|------------------|-------------|
| **NDCG@10** | 0.00267 | 0.00016 | **0.00960** | **3.6x** ↑ |
| **Recall@10** | 0.007 | 0.0004 | **0.02117** | **3.0x** ↑ |
| **Coverage** | 0.00004 | 0.00855 | **0.01500** | **375x** ↑ |

**Evaluation Details:** See `models/saved/baseline_evaluation_results.json`

## 📊 Key Features

### Two-Tower Architecture
```
User Tower:
  ├─ User ID Embedding (64D)
  ├─ GRU History Encoder (15 recent interactions)
  └─ Fusion Layer (residual connection)

Recipe Tower:
  ├─ Recipe ID Embedding (64D)
  ├─ Ingredient EmbeddingBag (Top 2000 vocab, mean pooling)
  ├─ Nutrition Dense (7 features → 64D)
  └─ Fusion: Concat(3×64D) → Dense(192→64) → ReLU → Dense(64)

Goal-Aware Re-ranking:
  └─ Boost scores for recipes fitting calorie budget
```

### Optimization Highlights
- ✅ **Mixed Precision (AMP)**: 10-15x speedup on RTX 3060
- ✅ **Gradient Clipping**: Prevents training instability
- ✅ **Temporal Split**: 80/20 train/test by date
- ✅ **Reduced History**: 15 items (30% faster, minimal quality loss)
- ✅ **AdamW + LR Scheduler**: Better convergence

## 🔬 Evaluation Scripts

```bash
# Evaluate baselines (Popularity + KNN)
python evaluate_baselines.py

# Evaluate Two-Tower model
python evaluate_twotower.py

# Full-corpus ranking evaluation
python evaluate_improved_model.py
```

## 📚 Documentation

- **`UI_IMPROVEMENTS.md`**: Detailed UI design documentation
- **`QUICK_START_UI.md`**: Step-by-step UI viewing guide
- **`RTX3060_OPTIMIZATIONS_APPLIED.md`**: GPU optimization details
- **`DATA_INSTRUCTIONS.md`**: Dataset setup instructions

## 🎓 Academic Context

This project was developed for a graduate-level Recommender Systems course, addressing:

1. ✅ **Complexity**: Two-Tower architecture with GRU history encoding
2. ✅ **Personalization**: User interaction history + collaborative filtering
3. ✅ **Content Features**: Ingredients + nutritional attributes
4. ✅ **Evaluation**: Proper temporal split, multiple metrics, baselines
5. ✅ **Goal-Awareness**: Calorie-budget re-ranking
6. ✅ **Production-Ready**: FastAPI backend, modern UI, GPU optimization

## 🚀 API Endpoints

### `POST /predict-food`
Upload food image for classification
```json
{
  "food_name": "pizza",
  "confidence": 0.85,
  "nutrition": {
    "calories": 285,
    "protein": 12,
    "carbs": 36,
    "fat": 10
  }
}
```

### `POST /recommend`
Get personalized recipe recommendations
```json
{
  "user_id": "user_1",
  "current_calories": 1500,
  "daily_goal": 2000
}
```

Returns: Top-10 recipes with budget indicators and nutritional info

## 🤝 Contributing

This is an academic project. For questions or collaboration:
- Open an issue for bugs/suggestions
- See evaluation results in `models/saved/`

## 📄 License

Academic project - TMU Recommender Systems Course (Fall 2025)

## 🙏 Acknowledgments

- **Datasets**: Food-101, Food.com (Kaggle)
- **Frameworks**: PyTorch, FastAPI, React
- **Course**: TMU Graduate Recommender Systems
