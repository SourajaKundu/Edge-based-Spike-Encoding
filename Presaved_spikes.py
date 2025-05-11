import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import snntorch as snn
import snntorch.spikegen as spikegen
import snntorch.functional as SF
from snntorch import surrogate
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 128
data_path = './data'
output_dir = './encoded_spikes'
num_classes = 10
num_steps = 25  # number of time steps
beta = 0.9  # decay rate
num_epochs = 10
lr = 2e-4
encoding_type = "latency"  # options: "none", "latency", "rate", "delta"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Define a surrogate gradient function
spike_grad = surrogate.fast_sigmoid(slope=25)

# Download and preprocess MNIST dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# For visualization purposes, we'll need non-normalized images too
transform_no_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Step 1: Download datasets
def download_datasets():
    train_dataset = torchvision.datasets.MNIST(
        data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        data_path, train=False, download=True, transform=transform)
    
    # Dataset without normalization for visualization
    test_dataset_no_norm = torchvision.datasets.MNIST(
        data_path, train=False, download=True, transform=transform_no_norm)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_no_norm = torch.utils.data.DataLoader(
        test_dataset_no_norm, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, test_loader_no_norm

# Step 2: Encode and save spike data
def encode_and_save_spike_data(dataset_loader, output_path, encoding_type="rate", num_steps=25, device='cpu'):
    """
    Convert image data to spike encodings and save to disk
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_spikes = []
    all_labels = []
    
    print(f"Encoding data with {encoding_type} encoding...")
    for i, (img, label) in enumerate(tqdm(dataset_loader)):
        # Move to device (useful if encoding on GPU)
        img = img.to(device)
        label = label.to(device)
        
        # Flatten images
        img = img.view(img.size(0), -1)  # Flatten to [batch_size, 784]
        
        # Apply appropriate spike encoding
        if encoding_type == "rate":
            # Rate encoding: Convert intensity to spike probability
            spikes = spikegen.rate(img, num_steps=num_steps, gain=0.25)
        elif encoding_type == "latency":
            # Latency encoding: Convert intensity to spike timing
            img_rescaled = (img - img.min()) / (img.max() - img.min())
            spikes = spikegen.latency(img_rescaled, num_steps=num_steps, tau=5, threshold=0.01, normalize=True, linear=True)
        elif encoding_type == "delta":
            # Delta encoding: Generate spikes for changes in intensity
            spikes = spikegen.delta(img, threshold=0.01)
            if len(spikes.shape) == 2:  # If it doesn't include time dimension
                spikes = spikes.unsqueeze(1).repeat(1, num_steps, 1)
                spikes = spikes.permute(1, 0, 2)
        else:  # "none" - repeat the same input for all time steps
            spikes = img.unsqueeze(1).repeat(1, num_steps, 1)
            spikes = spikes.permute(1, 0, 2)
        
        # Ensure time dimension is first: [num_steps, batch_size, 784]
        if spikes.shape[0] != num_steps:
            spikes = spikes.permute(1, 0, 2)
        
        # Move back to CPU if needed for saving
        spikes = spikes.cpu()
        label = label.cpu()
        
        all_spikes.append(spikes)
        all_labels.append(label)
    
    # Concatenate all batches
    all_spikes = torch.cat(all_spikes, dim=1)  # Concatenate along batch dimension
    all_labels = torch.cat(all_labels, dim=0)
    
    # Save as a single file
    print(f"Saving encoded data to {output_path}")
    torch.save({
        'spikes': all_spikes,
        'labels': all_labels,
        'encoding': encoding_type,
        'num_steps': num_steps
    }, output_path)
    
    return output_path

# Step 3: Create dataset class for spike data
class EncodedSpikeDataset(Dataset):
    def __init__(self, spike_file_path, device='cpu'):
        """
        Dataset for loading pre-encoded spike data
        
        Args:
            spike_file_path: Path to the saved spike data file
            device: Device to load the data on
        """
        print(f"Loading spike data from {spike_file_path}")
        data = torch.load(spike_file_path, map_location=device)
        
        self.spike_data = data['spikes']      # Shape: [num_steps, num_samples, 784]
        self.labels = data['labels']          # Shape: [num_samples]
        self.encoding_type = data['encoding'] # Encoding method used
        self.num_steps = data['num_steps']    # Number of time steps
        
        self.num_samples = self.labels.shape[0]
        
        print(f"Loaded {self.num_samples} samples with {self.num_steps} time steps")
        print(f"Spike data shape: {self.spike_data.shape}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get all time steps for this sample
        sample_spikes = self.spike_data[:, idx, :]  # Shape: [num_steps, 784]
        label = self.labels[idx]
        
        return sample_spikes, label

# Step 4: Function to create datasets and loaders
def create_spike_dataloaders(train_spike_path, test_spike_path, batch_size=128, device='cpu'):
    """
    Create DataLoaders for the spike-encoded datasets
    """
    # Create dataset objects
    train_dataset = EncodedSpikeDataset(train_spike_path, device)
    test_dataset = EncodedSpikeDataset(test_spike_path, device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=(device == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=(device == 'cuda')
    )
    
    return train_loader, test_loader

# Define the Spiking Neural Network
class SpikingNetwork(nn.Module):
    def __init__(self, beta=0.9, spike_grad=None, num_steps=25):
        super().__init__()
        
        self.num_steps = num_steps
        
        # Input layer: 28x28 = 784 neurons (MNIST image size)
        # Hidden layer: 200 neurons
        # Output layer: 10 neurons (for 10 MNIST classes)
        
        # Initialize layers
        self.fc1 = nn.Linear(784, 200)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(200, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    def forward(self, x):
        """
        Forward pass of the spiking network
        
        Args:
            x: Input spike data with shape [num_steps, batch_size, 784]
               or [batch_size, num_steps, 784] which will be transposed
        """
        # Ensure input has the right shape [num_steps, batch_size, features]
        if x.shape[0] != self.num_steps and x.shape[1] == self.num_steps:
            x = x.transpose(0, 1)
        
        # Initialize hidden states and output lists
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer spikes and membrane potentials
        spk2_rec = []
        mem2_rec = []
        
        # Process each time step
        for step in range(self.num_steps):
            # Extract current time step data
            current_input = x[step]  # Shape: [batch_size, 784]
            
            # Layer 1: Leaky Integrate and Fire
            cur1 = self.fc1(current_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2: Leaky Integrate and Fire
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Store outputs
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        # Stack outputs over time dimension
        return torch.stack(spk2_rec), torch.stack(mem2_rec)

# Training function
def train(net, train_loader, optimizer, loss_fn, epoch, device):
    net.train()
    losses = []
    
    for spikes, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # Move data to device
        spikes = spikes.to(device)
        targets = targets.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        spk_rec, mem_rec = net(spikes)
        
        # Calculate loss
        loss = loss_fn(spk_rec, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        
        # Update parameters
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)

# Testing function
def test(net, test_loader, device):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spikes, targets in test_loader:
            # Move data to device
            spikes = spikes.to(device)
            targets = targets.to(device)
            
            # Forward pass
            spk_rec, _ = net(spikes)
            
            # Calculate accuracy
            # Sum spikes over time steps for rate coding classification
            spk_sum = spk_rec.sum(dim=0)
            _, predicted = spk_sum.max(1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100 * correct / total

# Visualization function
def visualize_results(train_losses, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Download datasets
    train_loader, test_loader, test_loader_no_norm = download_datasets()
    
    # Define file paths for encoded data
    train_spike_path = os.path.join(output_dir, f"train_{encoding_type}_spikes.pth")
    test_spike_path = os.path.join(output_dir, f"test_{encoding_type}_spikes.pth")
    
    # Step 2: Check if encoded data exists, if not, create it
    if not os.path.exists(train_spike_path):
        print("Encoding training data...")
        encode_and_save_spike_data(
            train_loader, 
            train_spike_path, 
            encoding_type=encoding_type, 
            num_steps=num_steps,
            device=device
        )
    
    if not os.path.exists(test_spike_path):
        print("Encoding test data...")
        encode_and_save_spike_data(
            test_loader, 
            test_spike_path, 
            encoding_type=encoding_type, 
            num_steps=num_steps,
            device=device
        )
    
    # Step 3: Create spike dataloaders
    spike_train_loader, spike_test_loader = create_spike_dataloaders(
        train_spike_path, 
        test_spike_path, 
        batch_size=batch_size,
        device=device
    )
    
    # Step 4: Initialize network
    net = SpikingNetwork(beta=beta, spike_grad=spike_grad, num_steps=num_steps).to(device)
    print(net)
    
    # Step 5: Define optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = SF.ce_rate_loss()
    
    # Step 6: Train the network
    train_loss_history = []
    test_acc_history = []
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss = train(net, spike_train_loader, optimizer, loss_fn, epoch, device)
        train_loss_history.append(train_loss)
        
        # Test the model
        test_acc = test(net, spike_test_loader, device)
        test_acc_history.append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Step 7: Visualize results
    visualize_results(train_loss_history, test_acc_history)
    
    # Step 8: Save the model
    model_save_path = os.path.join(output_dir, f"snn_{encoding_type}_model.pth")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_history,
        'test_acc': test_acc_history,
        'num_steps': num_steps,
        'encoding_type': encoding_type,
        'beta': beta
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")
    print(f"Best accuracy: {max(test_acc_history):.2f}%")
