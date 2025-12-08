import requests
import os

def test_api():
    base_url = "http://localhost:8000"
    
    # 1. Test Root
    try:
        r = requests.get(f"{base_url}/")
        print("Root:", r.json())
    except Exception as e:
        print("Failed to connect to API:", e)
        return

    # 2. Test Recommendation
    payload = {
        "user_id": "user_1",
        "current_calories": 1500,
        "daily_goal": 2000
    }
    try:
        r = requests.post(f"{base_url}/recommend", json=payload)
        print("\nRecommendations:", r.json())
    except Exception as e:
        print("Recommendation failed:", e)

    # 3. Test Image Prediction
    # Use one of the mock images
    test_img_path = "data/mock_images/pizza/img_0.jpg"
    if os.path.exists(test_img_path):
        try:
            files = {'file': open(test_img_path, 'rb')}
            r = requests.post(f"{base_url}/predict-food", files=files)
            print("\nPrediction:", r.json())
        except Exception as e:
            print("Prediction failed:", e)
    else:
        print(f"Test image {test_img_path} not found.")

if __name__ == "__main__":
    test_api()
