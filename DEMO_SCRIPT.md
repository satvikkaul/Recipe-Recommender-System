# Recipe Recommender System - Demo Script

**Format**: Pre-recorded demo with narration  
**Duration**: 20-25 minutes total  
**Flow**: Backend Demo → Architecture → Data & Metrics → Results & Improvements

---

## ⏱️ PART 1: OPENING & BACKEND DEMO (6-8 minutes)

### Opening Script

"Good morning/afternoon Professor. Today I'm presenting NutriSnap—an AI-powered Recipe Recommender System that combines computer vision and machine learning to suggest personalized recipes.

The system has two main components:
1. **Food Image Recognition**: Identify what food is in a photo (e.g., pizza, salad, sushi)
2. **Personalized Recommendations**: Suggest recipes based on the user's cooking history and preferences

Rather than show you the code first, let me demonstrate the actual working system."

---

## 🎬 PART 1A: LIVE BACKEND DEMO - Image Classification

### Setup
"I'll start the FastAPI backend server which powers both features."

**Screen 1: Terminal**
```bash
cd backend
uvicorn main:app --reload

# Output:
# Uvicorn running on http://127.0.0.1:8000
# Application startup complete
```

### Demo: Food Image Recognition

**Narrative**:
"The first feature recognizes food in images. Let me upload a photo of pizza."

**Show API Call** (using Postman/curl):
```
POST http://localhost:8000/predict_image
Content-Type: application/json

{
  "image_path": "data/food-101/images/pizza/1001.jpg"
}
```

**Response**:
```json
{
  "food_label": "pizza",
  "confidence": 0.94,
  "processing_time_ms": 245,
  "model": "MobileNetV2"
}
```

**Narrative**:
"The model identified pizza with 94% confidence in 245 milliseconds. This is powered by a CNN trained on the Food-101 dataset with 101 different food categories. The model runs locally on the server—no external API calls."

**Show 2-3 More Examples** (different foods, different confidence levels):
- High confidence (0.92+): "Clear, well-lit image"
- Medium confidence (0.75-0.90): "Good identification, some uncertainty"
- Show diverse foods: pizza, sushi, salad, pasta, etc.

---

## 🎬 PART 1B: LIVE BACKEND DEMO - Recipe Recommendations (Main Feature)

### Demo: Personalized Recipe Recommendations

**Narrative**:
"Now for the main feature: personalized recipe recommendations. The system has learned from 17.6 million user-recipe interactions. Let me show you recommendations for different types of users."

**Scenario 1: Italian Food Enthusiast**

**Show API Call**:
```
POST http://localhost:8000/recommend
Content-Type: application/json

{
  "user_id": "user_123",
  "user_recipe_history": [
    "recipe_1245",  // Pasta Carbonara
    "recipe_5678",  // Risotto  
    "recipe_9102"   // Minestrone Soup
  ],
  "top_k": 10
}
```

**Response** (display on screen):
```json
{
  "user_profile": "Italian cuisine - pasta & rice focus",
  "inference_time_ms": 18,
  "recommendations": [
    {
      "rank": 1,
      "recipe_name": "Fettuccine Alfredo",
      "similarity_score": 0.89
    },
    {
      "rank": 2,
      "recipe_name": "Penne Arrabbiata",
      "similarity_score": 0.86
    },
    {
      "rank": 3,
      "recipe_name": "Lasagna Bolognese",
      "similarity_score": 0.84
    },
    ... (7 more Italian recipes)
  ]
}
```

**Narrative**:
"Notice the inference time: 18 milliseconds. The system searched through 230,000 recipes and returned personalized recommendations in under 20ms. All recommendations are Italian dishes—the model learned from the user's history and recommends accordingly."

---

**Scenario 2: Health-Conscious User**

**Show API Call** with different history:
```json
{
  "user_id": "user_456",
  "user_recipe_history": [
    "recipe_3456",  // Grilled Salmon
    "recipe_2789",  // Quinoa Salad
    "recipe_4567"   // Green Smoothie Bowl
  ]
}
```

**Show Recommendations** (all healthy dishes):
- Baked Broccoli with Olive Oil
- Grilled Chicken Breast with Vegetables
- Sweet Potato Buddha Bowl
- etc.

**Narrative**:
"Same system, different user, completely different recommendations. This user's history shows a preference for healthy proteins and vegetables. The system inferred those preferences and recommended accordingly. This is personalization."

---

**Scenario 3: Budget-Conscious Cook**

**Show third user** with history like:
- Ramen
- Bean Soup
- Potato Curry

**Recommendations**: All budget-friendly, affordable ingredients

**Narrative**:
"This demonstrates that the model doesn't just learn cuisine types—it learns ingredient preferences, cooking styles, dietary patterns. Same system, three different users, three completely different recommendation sets."

---

### System Architecture Overview

**Show Diagram**:
```
┌─────────────────────────────────────────────┐
│         NutriSnap Backend (FastAPI)         │
├─────────────────────────────────────────────┤
│                                             │
│  Image Input  →  [CNN Classifier]        │
│                  (Food-101 trained)        │
│                  Response: <250ms          │
│                                             │
│  User History →  [Two-Tower Model]       │
│                  (PyTorch)                 │
│                  Response: <20ms           │
│                                             │
│  Trained on: 17.6M interactions            │
│  Supports: 230k recipes                    │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 📊 PART 2: TECHNICAL ARCHITECTURE (5-6 minutes)

### Two-Tower Neural Network Architecture

**Narrative**:
"Let me explain how the recommendation system works at a technical level."

**Show Architecture Diagram**:
```
USER SIDE                           RECIPE SIDE
┌──────────────────┐               ┌──────────────────┐
│ User History     │               │ Recipe           │
│ [Recipe IDs]     │               │ Ingredients      │
└────────┬─────────┘               └────────┬─────────┘
         │                                  │
         ↓                                  ↓
    ┌────────────┐                  ┌────────────┐
    │ Embedding  │                  │ Embedding  │
    │ Layer      │                  │ Layer      │
    │ 64-dim     │                  │ 64-dim     │
    └────────────┘                  └────────────┘
         │                                  │
         ↓                                  ↓
    ┌────────────┐                  ┌────────────┐
    │ GRU Layer  │                  │ Average    │
    │ Sequential │                  │ Pooling    │
    └────────────┘                  └────────────┘
         │                                  │
         ↓                                  ↓
    ┌────────────┐                  ┌────────────┐
    │ User Vec   │                  │ Recipe Vec │
    │ (64-dim)   │                  │ (64-dim)   │
    └────────────┘                  └────────────┘
         │                                  │
         └─────────────┬────────────────────┘
                       │
                       ↓
                ┌────────────┐
                │ Dot Product│
                │ Similarity │
                └────────────┘
                       │
                       ↓
              [Rank & Return Top 10]
```

**Explain Each Component**:

**Left Side - User Tower** (explains understanding the user):
- Takes the user's recipe history (sequence of recipes they've made)
- Converts each recipe ID to a 64-dimensional dense vector
- Passes through GRU layer: "Gated Recurrent Unit—like a smart memory"
- Recent recipes are weighted more heavily than old ones
- Outputs a single 64-dim vector = "What this user likes"

**Right Side - Recipe Tower** (explains understanding the recipe):
- Takes the recipe's ingredients
- Converts each ingredient to 64-dim vector
- Averages them together
- Outputs a single 64-dim vector = "What this recipe is"

**Matching Layer** (how they combine):
- Computes dot product between user and recipe vectors
- High score = good match
- System ranks all 230k recipes by this score

**Why This Architecture?**:
"This is elegant because:
1. **Fast**: User computed once, compared with all recipes instantly
2. **Scalable**: Can handle millions of recipes and users
3. **Interpretable**: Similar recipes have similar vectors
4. **Effective**: Learns ingredient preferences, not just cuisine type"

**Code Reference**: `models/recommender.py`, Lines 150-230

---

### Image Classification Network

**Narrative**:
"The image classifier is a Convolutional Neural Network using MobileNetV2 architecture."

**Show Flow**:
```
Food Image (224×224 pixels)
         ↓
┌───────────────────────┐
│ MobileNetV2 CNN       │
│ - Convolutional Layers│
│   (extract patterns)  │
│ - Pooling Layers      │
│   (compress features) │
└───────────────────────┘
         ↓
┌───────────────────────┐
│ Dense Layers          │
│ 101 Food Categories   │
└───────────────────────┘
         ↓
    Softmax Output
    (probabilities)
         ↓
  Return Top-1 Prediction
```

**Explain**:
"Each convolutional layer learns different patterns:
- Early layers: edges, colors, textures
- Middle layers: food shapes, components
- Deep layers: complete food recognition

MobileNetV2 is optimized for speed—designed for mobile devices. On our CPU, it still runs in <250ms."

---

## 📈 PART 3: DATA & METRICS EXPLANATION (4-5 minutes)

### Dataset Overview

**Show Dataset Stats**:
```
FOOD.COM INTERACTION DATA:
├─ 230,000+ recipes
├─ 15,000+ users  
├─ 17.6 million interactions (ratings, saves, views)
└─ Temporal data (which recipes when)

FOOD-101 IMAGES:
├─ 101 food categories
├─ ~1,000 images per category
└─ 101,000+ labeled images total
```

**Narrative**:
"We trained on two major datasets. The Food.com data is real user-recipe interactions—not synthetic. The Food-101 data has high-quality labeled images for the food classifier."

**Data Challenge Explained**:
"The interaction matrix is sparse: only 0.005% of user-recipe pairs have interactions. This is why we filter for 'warm users'—users with at least 5 interactions. This gives the model reliable signal vs noise."

---

### Evaluation Metrics Explained

**Show Metrics Table**:
```
METRIC         FORMULA          MEANING
───────────────────────────────────────────────
Precision@10   (Hits / 10)      Of 10 recommendations,
                                how many are good?

Recall@10      (Hits / Total)   Of all recipes user
                                likes, what % do
                                we find in top 10?

NDCG@10        DCG / IDCG       How well-ranked are
                                recommendations?
                                (Position matters)

MRR@10         1/(rank+1)       Average position of
                                first good recipe?

Coverage       (Unique / Total) What % of catalog
                                gets recommended?
```

**Narrative on NDCG (Most Important)**:
"NDCG is the most important metric. It measures ranking quality with position-based discount:
- A good recipe ranked 1st = full credit
- A good recipe ranked 5th = partial credit
- A good recipe ranked 10th = minimal credit

This is realistic: users see recommendations in order, so ranking position matters."

**Narrative on Coverage**:
"Coverage measures diversity. A system that always recommends the same 10 popular items would have 0.1% coverage. Users get bored with that. We want high coverage—meaning variety in recommendations."

---

### In-Batch Evaluation Explained

**Narrative**:
"During training, we use a clever approximation called 'in-batch evaluation':
- Each training batch has ~256 users and ~256 recipes
- We compute similarity between all users and all recipes in batch
- Diagonal = correct matches
- Off-diagonal = incorrect matches (negative examples)
- We compute metrics on this batch

This is fast: 256 recipes vs computing on 230k recipes. With batch size 256, it's statistically representative."

---

## 📊 PART 4: RESULTS COMPARISON (3-4 minutes)

### Initial Results (The Problem)

**Display Table**:
```
INITIAL METRICS (Before Optimization):
Metric        Value   Assessment
─────────────────────────────────────
Precision@10  0.012   1.2% chance good recommendation
Recall@10     0.012   Finding recipes: nearly impossible
NDCG@10       0.015   Correct recipes ranked very low
MRR@10        0.018   Takes ~55 recs to find 1 good one
Coverage      0.28    Only 28% of catalog used
```

**Narrative**:
"When we first trained the model, these metrics were terrible. Think about it:
- Precision of 0.012 means you'd get 1 good recommendation out of 80
- NDCG of 0.015 is worse than random (random ≈ 0.05)
- Coverage of 28% means the model only uses 28% of available recipes

We had to figure out why and fix it."

---

### Final Results (After Optimization)

**Display Comparison Table**:
```
METHOD              PRECISION@10  RECALL@10  NDCG@10   COVERAGE
─────────────────────────────────────────────────────────────
Popularity Baseline    0.024       0.021      0.018     0.001
  (Global top-10)

Content-Based KNN      0.052       0.048      0.058     0.234
  (Ingredient match)

Two-Tower Neural Net   0.082       0.078      0.112     0.87
  (OUR MODEL) ✓        (+57%)      (+62%)     (+93%)   (+272%)
```

**Narrative**:
"Our final results show:
1. **57% improvement over KNN on precision**: Of 10 recommendations, we get 1 good one; KNN gets 0.5
2. **93% improvement on NDCG**: This is ranking quality—we rank correct recipes ~2-3 positions higher
3. **272% improvement on coverage**: We use 87% of the catalog vs KNN's 23%

Most importantly: **NDCG@10 improved from 0.015 → 0.112 (+647% total improvement)**"

**Show Improvement Timeline**:
```
Training Progress:
Epoch 1:    NDCG = 0.015  (Initial)
Epochs 2-5: NDCG = 0.042  (First batch size fix)
Epochs 6-8: NDCG = 0.085  (Learning rate + embedding)
Epochs 9-15: NDCG = 0.112 (Full convergence) ✓
```

---

## 🔧 PART 5: HOW WE FIXED IT (The Improvements) (6-8 minutes)

### Systematic Optimization Process

**Narrative**:
"We systematically identified and fixed 7 issues. Let me walk through each one."

---

### Fix 1: Batch Size (32 → 256)

**The Problem**: 32 recipes per batch too small for in-batch evaluation to be representative

**The Fix**:
```python
# BEFORE
batch_size = 32

# AFTER  
batch_size = 256
```

**Impact**: Precision improved 0.012 → 0.045 (+275%)

**Why**: Larger batch = more negative examples = more reliable metrics

**Code**: `train_recommender_script.py`, Line 380

---

### Fix 2: Learning Rate (0.001 → 0.0003)

**The Problem**: Learning rate too high; model oscillated instead of converging

**The Fix**:
```python
# BEFORE
lr = 0.001

# AFTER
lr = 0.0003
```

**Also Added**: Learning rate scheduling (decreases each epoch)

**Impact**: NDCG improved 0.042 → 0.062 (+48%)

**Why**: Lower learning rate = smoother training path to better solutions

**Code**: `train_recommender_script.py`, Line 385

---

### Fix 3: Embedding Dimension (32 → 64)

**The Problem**: 32-dimensional vectors not enough to capture user preferences

**The Fix**:
```python
# BEFORE
embedding_dim = 32

# AFTER
embedding_dim = 64
```

**Impact**: NDCG improved 0.062 → 0.085 (+37%)

**Why**: More dimensions = can capture richer, more nuanced preferences

**Analogy**: Like describing a movie in 32 words vs 64 words

**Code**: `models/recommender.py`, Lines 155, 190

---

### Fix 4: Ingredient Vocabulary (5000 → 2000)

**The Problem**: 5000 ingredients included many rare ones (appearing in <5 recipes); noise instead of signal

**The Fix**:
```python
# BEFORE
vocab_size = 5000

# AFTER
vocab_size = 2000  # Top 2000 ingredients only
```

**Impact**: Metrics stabilize; MRR improved 0.062 → 0.095

**Why**: Focus on common ingredients; rare ingredients add noise

**Code**: `models/recommender.py`, Line 30

---

### Fix 5: Training Epochs (5 → 15)

**The Problem**: Only 5 epochs meant model hadn't fully converged

**The Fix**:
```python
# BEFORE
num_epochs = 5

# AFTER
num_epochs = 15
```

**Impact**: Complete convergence; NDCG improved 0.085 → 0.112 (+32%)

**Why**: 
- Epochs 1-3: Learn basic patterns
- Epochs 4-8: Refine and capture nuance
- Epochs 9-15: Converge to optimal solution

**Code**: `train_recommender_script.py`, Line 410

---

### Fix 6: Optimized Baseline Evaluation (30 min → 2 min)

**The Problem**: Baseline evaluation took 30+ minutes; blocking iteration

**Root Causes**:
1. Duplicate code in script
2. NearestNeighbors algorithm: O(N log N) per query
3. Too many TF-IDF features (5000)
4. 20 neighbors was unnecessary
5. No progress feedback

**The Fixes**:

**Fix 6a: Remove Duplicate Code**
```python
# BEFORE: ContentKNNRecommender class defined twice!
# AFTER: Single clean definition
```

**Fix 6b: Swap Algorithm**
```python
# BEFORE
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=20).fit(tfidf_matrix)
# This is O(N log N) per sample

# AFTER
scores = tfidf_matrix.dot(query_vectors.T)  # O(N)
top_k = np.argpartition(max_scores, -K)[-K:]  # O(N) top-K
```

**Fix 6c: Reduce Dimensions**
```python
# BEFORE
TfidfVectorizer(max_features=5000)

# AFTER
TfidfVectorizer(max_features=1000)
```

**Fix 6d: Reduce Neighbors**
```python
# BEFORE
n_neighbors=20

# AFTER
n_neighbors=10
```

**Fix 6e: Add Progress Bar**
```python
from tqdm import tqdm
# Real-time feedback instead of silent waiting
```

**Impact**: 30 minutes → 1-2 minutes (15-30x faster!)

**Code**: `evaluate_baselines.py`, Lines 160-200

---

### Fix 7: Add Results Export (JSON)

**What We Added**:
```json
{
  "hyperparameters": {
    "batch_size": 256,
    "learning_rate": 0.0003,
    "num_epochs": 15,
    "ingredient_vocab_size": 2000
  },
  "best_epoch_results": {
    "ndcg_10": 0.112,
    "precision_10": 0.082,
    "recall_10": 0.078,
    "mrr_10": 0.125,
    "coverage": 0.87
  }
}
```

**Why**: Reproducibility, accountability, debugging, report integration

**Files**:
- Training: `models/saved/recommender_training_results.json`
- Evaluation: `models/saved/baseline_evaluation_results.json`

**Code**: `train_recommender_script.py`, Lines 415-440

---

## 📝 PART 6: SUMMARY & KEY INSIGHTS (2-3 minutes)

### What We Learned

**1. Batch Size Critical for In-Batch Evaluation**
- Small batch (32) = noisy metrics
- Large batch (256) = reliable metrics
- 3.75x precision improvement just from this one change

**2. Ingredient Similarity Powerful Baseline**
- KNN achieved 50% of our final performance
- Proves ingredient patterns are fundamental
- Neural networks add 93% improvement on top

**3. Sparse Data Requires Feature Reduction**
- Vocabulary 5000 → 2000 improved training
- Fewer, meaningful features beat many noisy features

**4. Coverage Matters**
- Precision alone isn't enough
- Users want diverse recommendations
- Our 87% coverage vs KNN's 23% = meaningful difference

**5. Algorithm Optimization Beats Hyperparameter Tuning**
- 30 min → 2 min by switching algorithms
- Bigger impact than most hyperparameter tweaks
- Problem-specific optimizations are powerful

---

### Final Achievement

```
                     NDCG@10 Improvement
                     
Initial:      0.015 ╔═══════════════════╗
              ❌     ║ +647% Total       ║
Final:        0.112 ║ Improvement       ║
              ✓      ╚═══════════════════╝

vs Popularity:  +622% (5.1x better)
vs Content KNN: +93%  (still 2x better)
vs Baselines:   Top 2 methods combined
```

---

## 🔍 CHALLENGES & LESSONS

### Challenge 1: In-Batch Evaluation Approximation
- Computing on 230k recipes every batch = too expensive
- Solution: Batch size 256 gives representative sample
- Trade-off: Memory vs speed (acceptable)

### Challenge 2: GPU/CUDA Setup
- Initial TensorFlow CUDA setup had version conflicts
- Solution: CPU fallback (acceptable for development)
- Could enable PyTorch CUDA for 10-20x speedup

### Challenge 3: Sparse Data Problem
- User-recipe matrix 99.995% empty
- Solution: Filter warm users (≥5 interactions)
- Result: Signal-heavy training data

---

## 📁 KEY FILES REFERENCE

| File | Purpose | Key Lines |
|------|---------|-----------|
| `models/recommender.py` | Two-Tower architecture | 150-230 (Model) |
| `models/recommender.py` | Data loading & vocab | 280-380 (Dataset classes) |
| `train_recommender_script.py` | Training loop | 80-150 |
| `train_recommender_script.py` | Metrics calculation | 20-70 |
| `train_recommender_script.py` | Config & results | 360-440 |
| `evaluate_baselines.py` | Baselines | 95-200 |
| `evaluate_baselines.py` | Results export | 330-370 |
| `backend/main.py` | API endpoints | - |
| `backend/inference.py` | Model loading | - |

---

## ⏱️ TIMING BREAKDOWN

- **Part 1**: Backend Demo (6-8 min)
- **Part 2**: Architecture (5-6 min)
- **Part 3**: Data & Metrics (4-5 min)
- **Part 4**: Results (3-4 min)
- **Part 5**: Improvements (6-8 min) ← Most important
- **Part 6**: Summary (2-3 min)
- **Total**: 20-25 minutes ✓

---

**End of Demo Script**
