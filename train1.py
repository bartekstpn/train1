# importy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ustawienie urządzenia > pierwsze kolejność karta graficzna jeśli nie - CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformacja
# zmniejszanie do 224x224px > Do tensora > normalizacja
img_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# przygotowanie danych > ładowanie > dostarczanie w partiach > mieszanie
train_set = datasets.ImageFolder('test_dataset', img_transforms)
batch_size = 12
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# ustawiamy kryterium strat
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

# przygotowanie modelu  > zamrażanie > ustawianie warstw > przenoszenie na urządzenia
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad=False

model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2)
)
model.to(device)

#optymalizator i współczynnik uczenia
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#trenowanie modelu
for epoch in range(10):
    train_loss = 0.0
    print('epoch', epoch)
    for inputs, target in train_loader:
        optimizer.zero_grad()
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss/len(train_loader)
    print('epoch: ', epoch+1, "train_loss: ", train_loss)