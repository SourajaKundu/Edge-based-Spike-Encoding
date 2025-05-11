import torch
import torch.nn as nn
import torchvision
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikegen

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
batch_size = 128
data_path = './data'
output_dir = './rate_encoding'
num_classes = 10
num_steps = 5  # number of time steps
beta = 0.9  # decay rate
num_epochs = 10
lr = 2e-4
encoding_type = "latency"  # options: "none", "latency", "rate", "delta"
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

# Define the Spiking Neural Network
class SpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer: 28x28 = 784 neurons (MNIST image size)
        # Hidden layer: 200 neurons
        # Output layer: 10 neurons (for 10 MNIST classes)
        
        # Initialize layers
        self.fc1 = nn.Linear(784, 200)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(200, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    def forward(self, x, num_steps):
        # Initialize hidden states and output lists
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer spikes
        spk2_rec = []
        mem2_rec = []
        
        #print("Shape of x (one sample)", x.shape)  
        # First, flatten the input images
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Shape: [batch_size, 784]
        
        # Apply spike encoding based on the selected method
        if encoding_type == "latency":
            # Latency encoding: Convert intensity to spike timing
            x_rescaled = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min())
            spike_data = spikegen.latency(x_rescaled, num_steps=num_steps, tau=5, threshold=0.01, normalize=True, linear=True)
            # spike_data shape: [batch_size, num_steps, 784]
        elif encoding_type == "rate":
            # Rate encoding: Convert intensity to spike probability
            spike_data = spikegen.rate(x_flat, num_steps=num_steps, gain=0.25)
            # spike_data shape: [batch_size, num_steps, 784]
        elif encoding_type == "delta":
            # Delta encoding: Generate spikes for changes in intensity
            spike_data = spikegen.delta(x_flat, threshold=0.01, off_spike=True)
            #print(f"Shape of delta modulation: {spike_data.shape}\n")
            
            if len(spike_data.shape) == 2:  # If it doesn't include time dimension
                spike_data = spike_data.unsqueeze(1).repeat(1, num_steps, 1)
                spike_data = spike_data.permute(1, 0, 2)
                #print(f"Shape of delta modulation after modulaion: {spike_data.shape}\n")
        else:  # "none" - repeat the same input for all time steps
            
            spike_data = x_flat.unsqueeze(1).repeat(1, num_steps, 1)
            spike_data = spike_data.permute(1, 0, 2)
            # spike_data shape: [batch_size, num_steps, 784]
        
        # Transpose to get time dimension first: [num_steps, batch_size, 784]
        #spike_data = spike_data.permute(1, 0, 2)
       
        
        
        # Process each time step
        for step in range(num_steps):
            # Extract current time step data
            #print(f"Shap[e of spike data: {spike_data.shape}\n")
            current_input = spike_data[step]  # Shape: [batch_size, 784]
            #print("Shape of data fed to net each time step", current_input.shape) 
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
    
# Initialize the network
net = SpikingNetwork()
print(net)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define loss function and optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_fn = SF.ce_rate_loss()

# Training function
def train(net, train_loader, optimizer, epoch):
    net.train()
    losses = []
    
    for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        data = data.to(device)
        targets = targets.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        spk_rec, mem_rec = net(data, num_steps)
        
        # Calculate loss
        loss = loss_fn(spk_rec, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)

# Testing function
def test(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            spk_rec, _ = net(data, num_steps)
            
            # Calculate accuracy
            # Sum spikes over time steps to get rate coding
            spk_sum = spk_rec.sum(dim=0)
            _, predicted = spk_sum.max(1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100 * correct / total

# Training loop
train_loss_history = []
test_acc_history = []

for epoch in range(num_epochs):
    train_loss = train(net, train_loader, optimizer, epoch)
    test_acc = test(net, test_loader)
    
    train_loss_history.append(train_loss)
    test_acc_history.append(test_acc)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Accuracy = {test_acc:.2f}%")
    
print(f"Max accuracy: {max(test_acc_history)}")

# Plot training results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(test_acc_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')

plt.tight_layout()
plt.show()

# Function to visualize spiking activity for a single example
def visualize_spikes(net, example_image):
    net.eval()
    
    # Prepare the input
    example_tensor = example_image.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        spk_rec, mem_rec = net(example_tensor, num_steps)
    
    # Plot spiking activity
    plt.figure(figsize=(15, 8))
    
    # Show the input image
    plt.subplot(1, 3, 1)
    plt.imshow(example_image.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot spike times for output neurons
    spike_data = spk_rec.cpu().numpy()
    
    plt.subplot(1, 3, 2)
    plt.imshow(spike_data[:, 0, :], aspect='auto', cmap='binary')
    plt.xlabel('Neuron Index')
    plt.ylabel('Time Step')
    plt.title('Output Spikes Over Time')
    
    # Plot membrane potentials for output neurons
    mem_data = mem_rec.cpu().numpy()
    
    plt.subplot(1, 3, 3)
    for i in range(10):
        plt.plot(mem_data[:, 0, i], label=f'Neuron {i}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Membrane Potential')
    plt.title('Membrane Potentials of Output Neurons')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Add function to visualize spike encoding for an example image
def visualize_spike_encoding(example_image):
    # Make sure input is on CPU
    example_image = example_image.cpu()
    
    # Create a normalized version between 0 and 1 for encoding
    x_norm = (example_image - example_image.min()) / (example_image.max() - example_image.min())
    
    # Flatten the image
    x_flat = x_norm.view(-1)  # Shape: [784]
    
    # Apply different encoding methods
    latency_encoded = spikegen.latency(x_flat, num_steps=num_steps, tau=5, threshold=0.01, normalize=True, linear=True)
    rate_encoded = spikegen.rate(x_flat, num_steps=num_steps, gain=0.25)
    
    # For delta encoding, we'll use a different approach to avoid issues
    # First create a sequence of gradually increasing values
    seq_steps = 5  # Number of steps for the sequence
    delta_input_seq = torch.stack([x_flat * (i/seq_steps) for i in range(seq_steps)])
    # Apply delta to consecutive frames
    delta_encoded = []
    for i in range(1, seq_steps):
        d = spikegen.delta(torch.stack([delta_input_seq[i-1], delta_input_seq[i]]), 
                           threshold=0.05, off_spike=False)
        delta_encoded.append(d[1])  # Take only the "change" frame
    
    # Convert to numpy for visualization
    latency_encoded = latency_encoded.view(num_steps, 28, 28).cpu().numpy()
    rate_encoded = rate_encoded.view(num_steps, 28, 28).cpu().numpy()
    if delta_encoded:
        delta_encoded_np = torch.stack(delta_encoded).view(-1, 28, 28).cpu().numpy()
    else:
        # Fallback if delta encoding fails
        delta_encoded_np = np.zeros((1, 28, 28))
    
    # Original image for reference
    img_original = example_image.squeeze().cpu().numpy()
    
    # Plot the encodings
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # Original image
    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Latency encoding samples
    for i in range(4):
        timestep = i * (num_steps // 4)
        axes[0, i+1].imshow(latency_encoded[timestep], cmap='binary')
        axes[0, i+1].set_title(f'Latency t={timestep}')
        axes[0, i+1].axis('off')
    
    # Rate encoding samples
    for i in range(4):
        timestep = i * (num_steps // 4)
        axes[1, i+1].imshow(rate_encoded[timestep], cmap='binary')
        axes[1, i+1].set_title(f'Rate t={timestep}')
        axes[1, i+1].axis('off')
    
    # Delta encoding samples
    for i in range(4):
        if i < len(delta_encoded_np):
            axes[2, i+1].imshow(delta_encoded_np[i], cmap='binary')
            axes[2, i+1].set_title(f'Delta t={i}')
        else:
            axes[2, i+1].imshow(np.zeros((28, 28)), cmap='binary')
            axes[2, i+1].set_title(f'Delta t={i}')
        axes[2, i+1].axis('off')
    
    # Set column labels
    axes[0, 0].set_title('Input Image')
    axes[1, 0].set_title('Encoding Types')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    
    plt.tight_layout()
    plt.show()

# Get a sample image for visualization
# Use the non-normalized version for visualization
example_image_no_norm, example_label = next(iter(test_loader_no_norm))
example_image_no_norm = example_image_no_norm[0]

# Get the normalized version for the actual network prediction
example_image, _ = next(iter(test_loader))
example_image = example_image[0]

# Visualize spike encoding methods
visualize_spike_encoding(example_image_no_norm)

# Visualize spiking activity with normalized image
visualize_spikes(net, example_image)

# Save the trained model
torch.save(net.state_dict(), 'snn_mnist_model.pth')

print("Training complete!")
