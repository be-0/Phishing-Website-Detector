# Hardware and user spercifications
import platform
import psutil
import getpass

# --- Setup: Imports ---
import os, seaborn, sklearn, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

# Set a random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# --- Data Loading and Preprocessing ---

# Load the dataset
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
phishing_websites = fetch_ucirepo(id=327) 
  
# data (as pandas dataframes) 
X = phishing_websites.data.features 
Y = phishing_websites.data.targets 

X = X.select_dtypes(include=[np.number])

# Map target labels: -1 -> 0, 0 -> 1, 1 -> 2 (Pytorch's CrossEntropyLoss does not accept negative targets)
Y = Y.replace({-1: 0, 0: 1, 1: 2})

# Split the data (80% train, 20% validation). Set random_state = random_seed.
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = random_seed)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
y_train_t = torch.tensor(Y_train.values, dtype=torch.long)
y_val_t = torch.tensor(Y_val.values, dtype=torch.long)

print(f"Training features shape: {X_train_t.shape}")
print(f"Validation features shape: {X_val_t.shape}")

# --- Model Definition ---
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(30, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)          
        self.fc3 = nn.Linear(16, 3)          
     
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


input_features = X_train_t.shape[1]
print(f"Model will accept {input_features} input features.")
print("--- 3. Model Class Defined ---")

def calculate_full_loss(model, criterion, X, y):
    """Helper function to calculate loss over an entire dataset."""
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        outputs = model(X)
        y = y.view(-1)
        loss = criterion(outputs, y)
    model.train() # Set model back to train mode
    return loss.item()

def train_with_minibatch(model, criterion, optimizer, X_train, y_train, X_val, y_val,
                         num_iterations, batch_size, check_every):

    train_losses = []
    val_losses = []
    iterations = []
    
    for i in range(1, num_iterations + 1):

        # Pick one random sample batch
        rand_index = np.random.choice(X_train.shape[0], batch_size, replace = False)
        rand_X_batch = X_train[rand_index]
        rand_y_batch = y_train[rand_index].long().view(-1)
        
        #Predict
        optimizer.zero_grad()
        outputs = model(rand_X_batch)
        
        #Compute loss
        loss = criterion(outputs, rand_y_batch)
        loss.backward() 
       
        #Adjust learning weights
        optimizer.step()

        #Check progress
        if i % check_every == 0:
            train_loss = calculate_full_loss(model, criterion, X_train, y_train)
            val_loss = calculate_full_loss(model, criterion, X_val, y_val)
    
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            iterations.append(i)

            print(f"Iteration {i}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")

            
    return train_losses, val_losses, iterations, model 


# --- Set Hyperparameters ---
LEARNING_RATE = 0.01
NUM_ITERATIONS = 3000
BATCH_SIZE = 32
CHECK_EVERY = round(NUM_ITERATIONS / 10)

# --- Model Initialization ---
model = Classifier() 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)

# --- Run Minibatch Training ---
train_losses, val_losses, iterations, model = train_with_minibatch(
    model, criterion, optimizer,
    X_train_t, y_train_t, X_val_t, y_val_t,
    NUM_ITERATIONS, BATCH_SIZE, CHECK_EVERY
)

# --- Plotting ---
print("Plotting...")
plt.figure(figsize=(14, 7))

plt.plot(iterations, train_losses, label='Train Loss', linestyle='-', color='green', marker='o')
plt.plot(iterations, val_losses, label='Validation Loss', linestyle='-', color='blue', marker='o')

plt.title('Loss plots')
plt.xlabel('Iterations')
plt.ylabel('Loss (CELoss)')
plt.legend()
plt.grid(True)
plt.show()

