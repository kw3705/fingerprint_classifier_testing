import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

"""
    Using the training dataset, train a model to recognize fingerprints using pytorch.
    Then, test the model using the testing dataset.
    The given dataset contains subfolders of class A, L, R, T, and W.
    Each subfolder contains images already of size 512x512.
    
    Code referenced from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
# Load the dataset
training_dataset = "C:\\Users\\Kevin\\PycharmProjects\\AI_Testing\\fingerprint_dataset_train"
testing_dataset = "C:\\Users\\Kevin\\PycharmProjects\\AI_Testing\\fingerprint_dataset_test"
train_dataset = torchvision.datasets.ImageFolder(
    root=training_dataset,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    num_workers=0,
    shuffle=True
)
test_dataset = torchvision.datasets.ImageFolder(
    root=testing_dataset,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    num_workers=0,
    shuffle=True
)
classes = ('A', 'L', 'R', 'T', 'W')

# Define the model for the dataset of 512x512 images
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
            running_loss = 0.0
print("Finished training")

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the test images: {100 * correct / total}%")

# Save the model
# PATH = './fingerprint_net.pth'