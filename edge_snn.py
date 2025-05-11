import os
import torch
import torch.nn as nn
import torchvision
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pywt

def wavelet_threshold_denoising(image, wavelet='rbio4.4', level=3): # Daubechies-1 wavelet

    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, detail_coeffs = coeffs[0], coeffs[1:]

    threshold_value = np.median(np.abs(detail_coeffs[-1])) / 0.6745 # Donoho's universal threshold

    def threshold(data, threshold_value):
        return np.sign(data) * np.maximum(np.abs(data) - threshold_value, 0) # Soft thresholding

    detail_coeffs = [(threshold(H, threshold_value), threshold(V, threshold_value), threshold(D, threshold_value))
                     for H, V, D in detail_coeffs]  # Noise removed

    denoised_image = pywt.waverec2([cA] + detail_coeffs, wavelet)
    return np.clip(denoised_image, 0, 255).astype(np.uint8)

def compute_gradient(image):
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction

def non_maximum_suppression(grad_mag, grad_dir):
    suppressed = np.zeros_like(grad_mag)
    angle = np.rad2deg(grad_dir) % 180

    for i in range(1, grad_mag.shape[0] - 1):
        for j in range(1, grad_mag.shape[1] - 1):
            q, r = 255, 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q, r = grad_mag[i, j + 1], grad_mag[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q, r = grad_mag[i + 1, j - 1], grad_mag[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q, r = grad_mag[i + 1, j], grad_mag[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q, r = grad_mag[i - 1, j - 1], grad_mag[i + 1, j + 1]

            if grad_mag[i, j] >= q and grad_mag[i, j] >= r:
                suppressed[i, j] = grad_mag[i, j]

    return suppressed

def modified_otsu_threshold(grad_mag):
    hist, bins = np.histogram(grad_mag.ravel(), bins=256, range=(0, 255))
    bins = bins[:-1]
    total_pixels = grad_mag.size
    max_variance = 0
    best_Tl, best_Th = 0, 0

    for Tl in range(1, 255):
        for Th in range(Tl+1, 255):
            w1, w2, w3 = np.sum(hist[:Tl]), np.sum(hist[Tl:Th]), np.sum(hist[Th:])
            if w1 == 0 or w2 == 0 or w3 == 0:
                continue

            p1 = np.sum(bins[:Tl] * hist[:Tl]) / w1
            p2 = np.sum(bins[Tl:Th] * hist[Tl:Th]) / w2
            p3 = np.sum(bins[Th:] * hist[Th:]) / w3
            u = p1 * w1 + p2 * w2 + p3 * w3

            sigma = w1 * (p1 - u) ** 2 + w2 * (p2 - u) ** 2 + w3 * (p3 - u) ** 2

            if sigma > max_variance:
                max_variance = sigma
                best_Tl, best_Th = Tl, Th

    return best_Tl, best_Th


def edge_tracking(weak_edges, strong_edges):
    final_edges = np.copy(strong_edges)
    rows, cols = weak_edges.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j] == 255 and np.any(strong_edges[i-1:i+2, j-1:j+2] == 255):
                final_edges[i, j] = 255

    return final_edges

def adaptive_edge_detection(image, sigma=0.8):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = wavelet_threshold_denoising(image)

    smoothed = cv2.GaussianBlur(denoised, (5, 5), sigma)

    magnitude, direction = compute_gradient(smoothed)

    suppressed = non_maximum_suppression(magnitude, direction)

    Tl, Th = modified_otsu_threshold(suppressed)

    strong_edges = (suppressed >= Th).astype(np.uint8) * 255
    weak_edges = ((suppressed >= Tl) & (suppressed < Th)).astype(np.uint8) * 255

    final_edges = edge_tracking(weak_edges, strong_edges)

    return final_edges

# Set random seed for reproducibility
torch.manual_seed(42)

# Define paths
original_mnist_dir = './mnist_png'
edge_mnist_dir = './mnist_edge'
original_cifar10_dir = './cifar_png'
edge_cifar10_dir = './cifar_edge'
os.makedirs(edge_cifar10_dir, exist_ok=True)

# Hyperparameters
batch_size = 128
num_classes = 10
beta = 0.9  # decay rate
num_epochs = 10
lr = 3e-4

# Step 1: Download MNIST dataset as PNG images
def download_mnist_as_png():
    # Download MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True)
    
    # Create directories for each digit
    for i in range(10):
        os.makedirs(os.path.join(original_mnist_dir, str(i)), exist_ok=True)
    
    # Save training images
    print("Saving training images as PNG...")
    for idx, (img, label) in enumerate(tqdm(train_dataset)):
        img_path = os.path.join(original_mnist_dir, str(label), f'train_{idx}.png')
        img.save(img_path)
    
    # Save test images
    print("Saving test images as PNG...")
    for idx, (img, label) in enumerate(tqdm(test_dataset)):
        img_path = os.path.join(original_mnist_dir, str(label), f'test_{idx}.png')
        img.save(img_path)
    
    print(f"Original MNIST images saved to {original_mnist_dir}")
    
    
# Step 1: Download CIFAR10 dataset as PNG images
def download_cifar_as_png():
    train_dataset = torchvision.datasets.CIFAR10(
        './data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        './data', train=False, download=True)
    
    
    for i in range(10):
        os.makedirs(os.path.join(original_cifar10_dir, str(i)), exist_ok=True)
    
    # Save training images
    print("Saving training images as PNG...")
    for idx, (img, label) in enumerate(tqdm(train_dataset)):
        img_path = os.path.join(original_cifar10_dir, str(label), f'train_{idx}.png')
        img.save(img_path)
    
    # Save test images
    print("Saving test images as PNG...")
    for idx, (img, label) in enumerate(tqdm(test_dataset)):
        img_path = os.path.join(original_cifar10_dir, str(label), f'test_{idx}.png')
        img.save(img_path)
    
    print(f"Original CIFAR10 images saved to {original_cifar10_dir}")    

# Step 2: Convert images to edge images using Canny edge detection
def convert_to_edge_images():
    # Create directories for each digit in edge folder
    for i in range(10):
        os.makedirs(os.path.join(edge_mnist_dir, str(i)), exist_ok=True)
        #os.makedirs(os.path.join(edge_cifar10_dir, str(i)), exist_ok=True)
    
    # Process all images
    print("Converting to edge images...")
    for digit in range(10):
        images = glob.glob(os.path.join(original_mnist_dir, str(digit), '*.png'))
        #images = glob.glob(os.path.join(original_cifar10_dir, str(digit), '*.png'))
        for img_path in tqdm(images, desc=f"Processing class {digit}"):
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            edges = adaptive_edge_detection(img, sigma=1.0)                        
            
            # Save edge image
            base_name = os.path.basename(img_path)
            edge_path = os.path.join(edge_cifar10_dir, str(digit), base_name)
            cv2.imwrite(edge_path, edges)
    
    print(f"Edge-detected images saved to {edge_cifar10_dir}")
    
    

# Step 3: Create a custom dataset for edge-detected images
class EdgeMNISTDataset(Dataset):
    def __init__(self, images_paths, labels):
        self.images_paths = images_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        binary_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to binary (0 for background, 1 for edges)
        # Assuming edges are white (255) in the edge images
        #binary_img = (img > 0).astype(np.float32)
        
        # Flatten the image (28x28 to 784)
        flat_img = binary_img.reshape(-1)
        
        # Convert to tensor
        tensor_img = torch.from_numpy(flat_img)
        
        return tensor_img, self.labels[idx]

# Step 4: Create train and test datasets
def create_datasets():
    all_images = []
    all_labels = []
    
    # Collect all images and their labels
    for digit in range(10):
        images = glob.glob(os.path.join(edge_mnist_dir, str(digit), '*.png'))
        all_images.extend(images)
        all_labels.extend([digit] * len(images))
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    
    # Convert labels to tensors
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create dataset objects
    train_dataset = EdgeMNISTDataset(X_train, y_train)
    test_dataset = EdgeMNISTDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Define the Spiking Neural Network (modified for direct spike input)
class EdgeSpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input layer: 28x28 = 784 neurons (MNIST image size)
        # Hidden layer: 200 neurons
        # Output layer: 10 neurons (for 10 MNIST classes)
        
        # Initialize layers
        self.fc1 = nn.Linear(784, 200)
        #self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.lif1 = snn.Leaky(beta=3, threshold=0.5)
        
        self.fc2 = nn.Linear(200, 10)
        #self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.lif2 = snn.Leaky(beta=3, threshold=0.5)
    
    def forward(self, x):
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # The input is already in binary spike format (0s and 1s)
        # Layer 1: Leaky Integrate and Fire
        x = x.to(torch.float32)
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        
        # Layer 2: Leaky Integrate and Fire
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        return spk2, mem2
    
class ComplexEdgeSpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
       
        
        # Increase hidden layer size
        self.fc1 = nn.Linear(784, 600)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.5)
        
        # Add a second hidden layer
        self.fc2 = nn.Linear(600, 400)
        self.lif2 = snn.Leaky(beta=0.7, threshold=0.3)
        
        # Add a third hidden layer
        self.fc3 = nn.Linear(400, 200)
        self.lif3 = snn.Leaky(beta=0.7, threshold=0.3)
        
        # Output layer
        self.fc4 = nn.Linear(200, 10)
        self.lif4 = snn.Leaky(beta=0.5, threshold=0.3)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        x = x.to(torch.float32)
        # Normalize input (important for edge data)
        if x.max() > 1.0:
            x = x / 255.0
       
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        spk1 = self.dropout(spk1)
      
        cur2 = self.fc2(spk1) 
        spk2, mem2 = self.lif2(cur2, mem2)
        spk2 = self.dropout(spk2)
        
        cur3 = self.fc3(spk2) 
        spk3, mem3 = self.lif3(cur3, mem3)
        spk3 = self.dropout(spk3)
        
        cur4 = self.fc4(spk3)
        spk4, mem4 = self.lif4(cur4, mem4)
 
        return spk4, mem4
    
class ResidualSpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 600)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.5)

        self.fc2 = nn.Linear(600, 600)
        self.lif2 = snn.Leaky(beta=0.7, threshold=0.3)

        self.fc3 = nn.Linear(600, 200)
        self.lif3 = snn.Leaky(beta=0.6, threshold=0.3)

        self.fc4 = nn.Linear(200, 10)
        self.lif4 = snn.Leaky(beta=0.5, threshold=0.3)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        x = x.to(torch.float32)
        if x.max() > 1.0:
            x = x / 255.0

        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        spk1 = self.dropout(spk1)

        # Skip connection around second block
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        spk2 = spk2 + spk1  # Skip connection
        spk2 = self.dropout(spk2)

        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)
        #spk3 = self.dropout(spk3)

        cur4 = self.fc4(spk3)
        spk4, mem4 = self.lif4(cur4, mem4)
        
        return spk4, mem4

class AttentionSpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(784, 600)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.5)

        self.attn_fc_q = nn.Linear(600, 64)
        self.attn_fc_k = nn.Linear(600, 64)
        self.attn_fc_v = nn.Linear(600, 600)

        self.fc2 = nn.Linear(600, 200)
        self.lif2 = snn.Leaky(beta=0.6, threshold=0.3)

        self.fc3 = nn.Linear(200, 10)
        self.lif3 = snn.Leaky(beta=0.5, threshold=0.3)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        x = x.to(torch.float32)
        if x.max() > 1.0:
            x = x / 255.0

        embed = self.embed(x)
        spk1, mem1 = self.lif1(embed, mem1)

        # Apply attention on spike outputs
        Q = self.attn_fc_q(spk1)
        K = self.attn_fc_k(spk1)
        V = self.attn_fc_v(spk1)

        attn_weights = torch.softmax(Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5), dim=-1)
        attn_out = attn_weights @ V

        spk2_in = attn_out + spk1  # residual connection
        cur2 = self.fc2(spk2_in)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3

    
class ImprovedEdgeSpikingNetwork(nn.Module):
    def __init__(self, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        
        # Increase hidden layer size
        self.fc1 = nn.Linear(784, 400)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.5)
        
        # Add a second hidden layer
        self.fc2 = nn.Linear(400, 200)
        self.lif2 = snn.Leaky(beta=0.7, threshold=0.4)
        
        # Output layer
        self.fc3 = nn.Linear(200, 10)
        self.lif3 = snn.Leaky(beta=0.5, threshold=0.3)
        
        # Recurrent connections (optional)
        self.rec1 = nn.Linear(400, 400)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Store spikes for all timesteps
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        
        # Convert input to float
        x = x.float()
        
        # Normalize input (important for edge data)
        if x.max() > 1.0:
            x = x / 255.0
        #print("Shape of data fed to net each time step", x.shape)  
        # Run network for multiple timesteps
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout(spk1)
            spk1_rec.append(spk1)
            
            # Add recurrent connection (optional)
            # This allows temporal information to propagate
            # x = x + 0.1 * self.rec1(spk1)
            
            cur2 = self.fc2(spk1) 
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout(spk2)
            spk2_rec.append(spk2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        
        # Convert spike recordings to tensors
        spk1_rec = torch.stack(spk1_rec, dim=0)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        spk3_rec = torch.stack(spk3_rec, dim=0)
        
        # Return both final membrane potential and accumulated spikes
        # Both can be used for classification
        return spk3_rec.sum(0), mem3#, spk3_rec.sum(0)

def temporal_cross_entropy_loss(spk_sum, mem_out, targets, alpha=0.5):
    """
    Combined loss that considers both membrane potential and spike count
    """
    # Membrane potential loss at final timestep
    mem_loss = nn.functional.cross_entropy(mem_out, targets)
    
    # Accumulated spike loss over all timesteps
    spk_loss = nn.functional.cross_entropy(spk_sum, targets)
    
    # Weighted combination
    return alpha * mem_loss + (1-alpha) * spk_loss

# Training function
def train(net, train_loader, optimizer, epoch):
    net.train()
    losses = []
    
    for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        data = data.to(device)
        targets = targets.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass (no time dimension now)
        spk, mem = net(data)
        
        # Calculate loss - using membrane potential for prediction
        #loss = nn.functional.cross_entropy(mem, targets)
        loss = temporal_cross_entropy_loss(spk, mem, targets)
        
        # Backward pass
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        
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
            _, mem = net(data)
            
            # Calculate accuracy
            _, predicted = mem.max(1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100 * correct / total

# Visualize function to display original, edge images and predictions
def visualize_examples(net, test_loader, num_examples=5):
    # Get a batch of data
    data, targets = next(iter(test_loader))
    
    # Select a few examples
    for i in range(min(num_examples, len(data))):
        input_data = data[i].to(device)
        target = targets[i].item()
        
        # Make prediction
        with torch.no_grad():
            _, mem = net(input_data.unsqueeze(0))
            _, predicted = mem.max(1)
            pred = predicted.item()
        
        # Reshape the flattened data back to 28x28 for visualization
        edge_img = input_data.cpu().numpy().reshape(28, 28)
        
        # Display
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(edge_img, cmap='binary')
        plt.title(f"Edge Image (Digit: {target})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(10), mem[0].cpu().numpy())
        plt.xlabel('Digit Class')
        plt.ylabel('Neuron Membrane Potential')
        plt.title(f"Network Output (Predicted: {pred})")
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # Execute steps only if files don't already exist
    if not os.path.exists(original_mnist_dir):
        download_mnist_as_png()
    
    if not os.path.exists(edge_mnist_dir) or len(os.listdir(edge_mnist_dir)) == 0:
        convert_to_edge_images()
    
    # Create datasets and loaders
    train_loader, test_loader = create_datasets()
    
    # Initialize the network
    net = ResidualSpikingNetwork()
    print(net)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Define optimizer
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay = 0.00001, momentum=0.00001)
    
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
    
    # Visualize some examples
    visualize_examples(net, test_loader)
    
    # Save the trained model
    torch.save(net.state_dict(), 'edge_snn_mnist_model.pth')
    
    print("Training complete!")
