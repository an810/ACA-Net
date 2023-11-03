import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Define your ACANet model and hyperparameters
per = ACANet(
    ch_in=80,
    latent_dim=192,
    embed_dim=256,
    embed_reps=2,
    attn_mlp_dim=256,
    trnfr_mlp_dim=256,
    trnfr_heads=8,
    dropout=0.2,
    trnfr_layers=3,
    n_blocks=2,
    max_len=10000,
    final_layer='1dE'
)

# Define the loss function (e.g., Mean Squared Error) and optimizer (e.g., Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(per.parameters(), lr=0.001)

# Generate some example data (you should replace this with your dataset loading code)
# Here we generate random input and target tensors for demonstration purposes.
# In practice, you should load your data using a DataLoader.
data_size = 1000
input_data = torch.randn(data_size, 999, 80)
target_data = torch.randn(data_size, 192, 999)  # Modify the target shape according to your task.

# Split the data into training and validation sets
train_input, val_input, train_target, val_target = train_test_split(input_data, target_data, test_size=0.2)

# Create DataLoader for training and validation data
batch_size = 32
train_dataset = TensorDataset(train_input, train_target)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_input, val_target)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    per.train()  # Set the model to training mode
    total_loss = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = per(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the model's parameters
        total_loss += loss.item()

    # Print the average training loss for this epoch
    print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")

    # Validation loop
    per.eval()  # Set the model to evaluation mode
    total_val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            val_outputs = per(inputs)
            val_loss = criterion(val_outputs, targets)
            total_val_loss += val_loss.item()

    # Print the average validation loss for this epoch
    print(f"Epoch {epoch + 1}, Validation Loss: {total_val_loss / len(val_loader)}")

# Save the trained model
torch.save(per.state_dict(), 'acanet_model.pth')
