from food_dataset import Food11Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model import NN
from ResNet import ResNetFood
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import numpy as np
from train import class_names
import matplotlib.pyplot as plt
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def predict_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)   

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        pred_prob = probs[0, pred_idx].item()

    return image, pred_class, pred_prob, probs[0].cpu()

def main(model):
    
    evaluation_dataset = Food11Dataset("evaluation", transform=transform)
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in evaluation_loader:
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
    
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Evaluation accuracy: {acc:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.show()
    
    # predict custom images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, path in zip(axes, ["test_images/im_test1.png", "test_images/im_test2.png"]):
        image, pred_class, pred_prob, probs = predict_image(path)
        ax.imshow(image)
        ax.set_title(f"{pred_class}\n({pred_prob:.2%})", fontsize=12)
        ax.axis("off")
    
        print(probs)

    plt.tight_layout()
    plt.savefig("images/predictions_side_by_side.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # evaluate created model
    '''model = NN()
    model.load_state_dict(torch.load('weights/weights.pth', weights_only=True))'''
    
    # evaluate post trained ResNet
    model = ResNetFood()
    model.load_state_dict(torch.load('weights/weights_res.pth', weights_only=True))
    main(model) 