import os
# We don't need legacy keras for standard CNN, usually. 
# But mixing might be safe. Let's try standard first.
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# Add root to path
sys.path.append(os.getcwd())

from models.image_classifier import build_custom_cnn, load_data

def main():
    print("TF Version:", tf.__version__)
    
    # Configuration
    DATA_DIR = "data/food-101/images"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 5  # Start with 5 for a quick real test, user can increase
    
    # Verify data exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} does not exist. Please ensure 'data/food-101/images' is populated.")
        return
    
    # Verify data exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} does not exist.")
        return

    try:
        train_ds, val_ds = load_data(DATA_DIR)
        class_names = train_ds.class_names
        print("Classes:", class_names)
        
        num_classes = len(class_names)
        model = build_custom_cnn(num_classes)
        model.summary()
        
        epochs = 3
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        
        model_save_path = 'models/saved/image_model.keras'
        # Save as native Keras format
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    except Exception as e:
        print("An error occurred during training:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
