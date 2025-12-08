import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(num_classes, img_height=224, img_width=224):
    """
    Builds a custom CNN model from scratch.
    """
    model = models.Sequential([
        # Rescaling
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # Dropout for regularization
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model

def load_data(data_dir, img_height=224, img_width=224, batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    return train_ds, val_ds

if __name__ == "__main__":
    pass
