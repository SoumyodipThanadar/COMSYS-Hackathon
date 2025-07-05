import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ===== CONFIG =====
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ===== TASK A: Gender Classification CNN =====
class GenderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ===== TASK B: Face Recognition CNN =====
class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ===== TASK A: Train Gender Classifier =====
def train_gender_classifier(train_dir, val_dir, model_path):
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GenderCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0, 0

        for x, y in tqdm(train_loader, desc=f"[Task A] Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.float().view(-1, 1).to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (output > 0.5).eq(y).sum().item()

        train_acc = train_correct / len(train_ds)

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.float().view(-1, 1).to(device)
                output = model(x)
                preds = (output > 0.5).float()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"📊 [Task A] Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, F1={f1:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Gender model saved to: {model_path}")

# ===== TASK A: Predict Gender =====
def predict_gender(model_path, test_folder, output_csv):
    model = GenderCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    for filename in os.listdir(test_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(test_folder, filename)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = (output > 0.5).float().item()
            gender = "Male" if pred == 1.0 else "Female"
            results.append((filename, gender))

    df = pd.DataFrame(results, columns=["filename", "gender"])
    df.to_csv(output_csv, index=False)
    print(f"📝 Gender predictions saved to {output_csv}")

# ===== TASK B: Train Face Recognition Model =====
def train_face_recognition(train_dir, val_dir, model_path):
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = FaceRecognitionCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0, 0

        for x, y in tqdm(train_loader, desc=f"[Task B] Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(output, dim=1)
            correct += preds.eq(y).sum().item()

        train_acc = correct / len(train_ds)

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                preds = torch.argmax(output, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f"📊 [Task B] Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, F1={f1:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Face recognition model saved to: {model_path}")

# ===== TASK B: Predict Identity =====
def predict_identity(model_path, test_folder, class_names, output_csv):
    num_classes = len(class_names)
    model = FaceRecognitionCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    for filename in os.listdir(test_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(test_folder, filename)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, dim=1).item()
            identity = class_names[pred]
            results.append((filename, identity))

    df = pd.DataFrame(results, columns=["filename", "identity"])
    df.to_csv(output_csv, index=False)
    print(f"📝 Identity predictions saved to {output_csv}")

# ===== MAIN =====
if __name__ == "__main__":
    # ✅ Task A: Gender Classification
    train_gender_classifier("Task_A/train", "Task_A/val", "gender_model.pth")
    predict_gender("gender_model.pth", "Task_A/test", "gender_predictions.csv")

    # ✅ Task B: Face Recognition
    train_face_recognition("Task_B/train", "Task_B/val", "face_model.pth")
    class_names = sorted(os.listdir("Task_B/train"))
    predict_identity("face_model.pth", "Task_B/test", class_names, "identity_predictions.csv")
