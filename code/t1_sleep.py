import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import argparse


def prepare_sleep_prediction_data(activity_df, sleep_df):
    """
    Prepare features and target for sleep prediction
    """
    # Merge activity and sleep data
    df = activity_df.merge(sleep_df, left_on=['Id', 'ActivityDate'], right_on=['Id', 'SleepDay'], how='inner')
    
    # Select features to use
    features = [
        'TotalSteps', 'TotalDistance',
        'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance',
        'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes'
    ]
    
    # Select target to predict
    target = ['TotalMinutesAsleep']
    
    # Split features and target
    X = df[features].values
    y = df[target].values
    
    return X, y

def predict_sleep(model, scaler, features):
    """
    Make predictions using the trained model
    """
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Convert to tensor and predict
    # TODO
    
    return prediction

class FitbitSleepPredictor(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the FitbitSleepPredictor neural network.

        Args:
            input_size (int): The number of input features.
        """
        super(FitbitSleepPredictor, self).__init__()

        # Define the neural network architecture using a Sequential container
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor containing the features.

        Returns:
            torch.Tensor: The predicted sleep duration.
        """
        # Pass the input through the entire network and return the result
        return self.network(x)


def train_sleep_predictor(X, y, epochs=300, batch_size=32, learning_rate=0.001):
    """
    Train a neural network model to predict sleep duration based on Fitbit activity data.

    Args:
        X (numpy.ndarray): Input features (activity data)
        y (numpy.ndarray): Target values (sleep duration)
        epochs (int): Number of training epochs
        batch_size (int): Size of mini-batches for training

    Returns:
        tuple: Trained model and feature scaler
    """
    # Split the data into training and testing sets
    # 80% for training, 20% for testing
    # TODO

    # Initialize and fit the StandardScaler to normalize the feature data
    # TODO

    # Fit the scaler to the training data and transform it
    # TODO
    
    # Transform the test data using the fitted scaler
    # TODO
    
    # Convert numpy arrays to PyTorch tensors for use with PyTorch models
    # TODO
    
    
    # Create an instance of the FitbitSleepPredictor model
    # The input size is set to the number of features in the training data
    model = FitbitSleepPredictor(input_size=X_train.shape[1])
    
    # Define the loss function (Mean Squared Error for regression tasks)
    # TODO
    
    # Define the optimizer (Adam) with a learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        # Set the model to training mode (enables dropout, etc.)
        model.train()
        epoch_train_loss = 0

        # Mini-batch training
        
        for i in range(0, len(X_train), batch_size):
            # Extract a mini-batch of data
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Forward pass: compute the model output for this batch
            outputs = model(batch_X)
            
            # Compute the loss between the predicted and actual values
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            # Clear previous gradients
            # This is necessary because PyTorch accumulates gradients by default.
            # If we don't clear them, the gradients from the previous batch would be
            # added to the current batch's gradients, leading to incorrect updates.
            # By zeroing the gradients, we ensure that each batch starts with a clean slate.
            optimizer.zero_grad()

            loss.backward()        # Compute gradients
            optimizer.step()       # Update model parameters

            epoch_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / (len(X_train) / batch_size)
        train_losses.append(avg_train_loss)
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test)
            val_loss = criterion(val_predictions, y_test)
            val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:                
            # Print the current epoch and test loss
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    # Return the trained model and the scaler for later use in predictions
    return model, scaler

def evaluate_model(model, X_test, y_test):
    model.eval()

    raise RuntimeError("Not implemented")
    # TODO


if __name__ == "__main__":

  # Set up argument parser
  parser = argparse.ArgumentParser(description='Train and evaluate a sleep prediction model')
  parser.add_argument('--datadir', type=str, default='../archive/clean', help='Directory containing the data files')
  parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
  parser.add_argument('--batch_size', type=int, default=32, help='Size of mini-batches for training')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
  parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
  parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
  args = parser.parse_args()

# Load your data
  activity_df = pd.read_csv(os.path.join(args.datadir, 'dailyActivity_merged.csv'))
  sleep_df = pd.read_csv(os.path.join(args.datadir, 'sleepDay_merged.csv'))

  # Prepare data
  X, y = prepare_sleep_prediction_data(activity_df, sleep_df)

  # Train model
  model, scaler = train_sleep_predictor(X, y, epochs=args.epochs, batch_size=args.batch_size)

  # Evaluate model
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
  X_test = torch.FloatTensor(scaler.transform(X_test))
  y_test = torch.FloatTensor(y_test)
  evaluate_model(model, X_test, y_test)

  # Make predictions
  new_features = np.array([[10000, 8.0, 2.0, 3.0, 3.0, 30, 45, 180]])  # Example input
  predicted_sleep = predict_sleep(model, scaler, new_features)
  print(f"Predicted sleep duration: {predicted_sleep[0][0]:.2f} minutes")

