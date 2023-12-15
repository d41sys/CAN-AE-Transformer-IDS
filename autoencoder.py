import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ExponentialLR
import os
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nodes):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, nodes),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(nodes, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, nodes),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(nodes, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create synthetic data for demonstration
# Replace this with your actual data
X_train = torch.randn(1000, 30)
input_dim = X_train.size(1)
latent_dim = 20
nodes = 128

# DataLoader setup
dataset = TensorDataset(X_train)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize model, loss, and optimizer
model = AutoEncoder(input_dim, latent_dim, nodes)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Initialize variables for Early Stopping
best_val_loss = float('inf')
patience, trials = 10, 0

train_loss, val_loss = [], []

# Training Loop
for epoch in range(100):
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        inputs, = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    scheduler.step()

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, = batch
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_val_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_val_loss = running_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}")

    train_loss.append(epoch_train_loss)
    val_loss.append(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        trials = 0
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "autoencoder_best_model.pth")
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping")
            break

# Plotting loss
plt.plot(train_loss[1:], 'b', label='Train')
plt.plot(val_loss[1:], 'r', label='Validation')
plt.title("Loss vs. Epoch for: Nodes=%i" % nodes)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()
