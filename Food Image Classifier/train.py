import os
from model import NN
from food_dataset import Food11Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from ResNet import ResNetFood

class_names = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles/Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable/Fruit"
}

def show_one_example_per_class(dataset):
    first_example_for_class = {}

    for fname in dataset.files:
        label = int(fname.split("_")[0])
        if label not in first_example_for_class:
            first_example_for_class[label] = fname

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for label in range(11):
        ax = axes[label]

        fname = first_example_for_class[label]
        img_path = os.path.join(dataset.image_path, fname)
        image = Image.open(img_path).convert("RGB")

        ax.imshow(image)
        ax.set_title(class_names[label], fontsize=12)
        ax.axis("off")

    axes[11].axis("off")

def display_training_results(train_losses, val_losses, train_accs, val_accs):
    
    plt.title('Loss')
    plt.plot(train_losses, label = 'loss')
    plt.plot(val_losses, label = 'validation loss')
    plt.legend()
    plt.savefig('images/loss.png')
    plt.show()

    plt.title('Accuracy')
    plt.plot(train_accs, label = 'acc')
    plt.plot(val_accs, label = 'validation acc')
    plt.legend()
    plt.savefig('images/accuracy.png')
    plt.show()

def main(model, criterion, optimizer):
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    train_dataset = Food11Dataset("training", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    show_one_example_per_class(train_dataset)

    validation_dataset = Food11Dataset("validation", transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    x = torch.zeros((1, 3, 512, 512))
    y = model(x)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(10):
        # -------------------
        # Training
        # -------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
        train_loss = running_loss / total
        train_acc = correct / total
    
        train_losses.append(train_loss)
        train_accs.append(train_acc)
    
        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
    
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
    
                outputs = model(images)
                loss = criterion(outputs, labels)
    
                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
    
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
    
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
    torch.save(model.state_dict(), 'weights/weights.pth')
    display_training_results(train_losses, val_losses, train_accs, val_accs)
    
if __name__ == '__main__':
    
    # Training of my model
    '''model = NN(input_dims=(3, 512, 512), num_classes=11)

    x = torch.zeros((1, 3, 512, 512))
    y = model(x)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)'''
    
    # Post Training of the pretrained ResNet
    model = ResNetFood()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    main(model, criterion, optimizer) 
