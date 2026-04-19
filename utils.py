import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image, model):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    classes = ["Benign", "Malignant"]

    return classes[pred.item()], confidence.item()