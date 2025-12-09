import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm

# Add root to path
sys.path.append(os.getcwd())

from models.image_classifier import build_model, get_data_loaders

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress Bar
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True)
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update Progress Bar
        loop.set_postfix(loss=loss.item(), acc=correct/total)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def main():
    # Setup Device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("\n" + "="*50)
        print("WARNING: CUDA (GPU) NOT DETECTED!")
        print("Training will be slow. Please ensure you installed PyTorch with CUDA support.")
        print("Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print(f"SUCCESS: CUDA DETECTED! Using {torch.cuda.get_device_name(0)}")
        print("="*50 + "\n")

    # Config
    DATA_DIR = "data/food-101/images"
    BATCH_SIZE = 32
    EPOCHS = 5
    MODEL_SAVE_PATH = "models/saved/image_model_pytorch.pth"
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return
        
    # Load Data
    print("Loading data...")
    train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    print(f"Found {len(class_names)} classes.")
    
    # Build Model
    model = build_model(len(class_names), device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, EPOCHS)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS} Result | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
              
    # Save Model
    print("Saving model...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # We save the state dict and also the class names so inference can use them
    save_dict = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }
    torch.save(save_dict, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
