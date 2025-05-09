import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random

with open('./normal.pickle', 'rb') as fp:
    normal = pickle.load(fp)

normal = normal / 3.5
normal = normal.reshape((30000, 10, 1080))
for i in range(30000):
    rnd = random.randint(1, 3)
    rnd_pos = random.randint(0, 1070)
    for s in range(10):
        normal[i][s][rnd_pos:rnd_pos + rnd] = 100.0

normal_label = np.zeros((30000,), dtype=int)

with open('./anomaly.pickle', 'rb') as fp:
    anomalies = pickle.load(fp)

anomalies = anomalies.reshape((30000, 10, 1080))
anomalies = anomalies / 3.5
anomalies_label = np.ones((30000,), dtype=int)

for i in range(30000):
    rnd = random.randint(1, 3)
    rnd_pos = random.randint(0, 1040)
    for s in range(10):
        anomalies[i][s][rnd_pos:rnd_pos + rnd] = 100.0

features = np.concatenate((normal, anomalies), axis=0)
labels = np.concatenate((normal_label, anomalies_label), axis=0)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

batch_size = 64
learning_rate = 0.001
num_epochs = 50

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10 * 1080, 5 * 1080)
        self.fc2 = nn.Linear(5 * 1080, 1080)
        self.fc3 = nn.Linear(1080, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

lowest_loss = 1000
# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss
        running_loss += loss.item()

    # Print the average loss for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    if running_loss < lowest_loss:
        lowest_loss = running_loss
        PATH = './best_anomaly_detector_50.pth'
        torch.save(model.state_dict(), PATH)

print('Finished Training')

PATH = './anomaly_detector_50.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    model.eval()
    X_test = X_test.to('cuda')
    y_test = y_test.to('cuda')
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy}")
