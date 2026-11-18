import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import argparse

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class ActivityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ActivityClassifier(nn.Module):
    def __init__(self, input_size=3):
        super(ActivityClassifier, self).__init__()
        #TODO
    
    def forward(self, x):
        return self.network(x)

def prepare_data(df):
    # Extract relevant features from the DataFrame
    # We're using 'TotalSteps', 'TotalDistance', and 'VeryActiveMinutes' as our input features
    X = df[['TotalSteps', 'TotalDistance', 'VeryActiveMinutes']].values
    
    # Create labels based on activity intensity
    # We'll categorize each day's activity into one of three levels:
    # 'very active', 'moderately active', or 'lightly active'
    conditions = [
        # Condition for 'very active':
        # VeryActiveMinutes is greater than 30 or TotalSteps is greater than 10000
        (df['VeryActiveMinutes'] > 30) | (df['TotalSteps'] > 10000),
        
        # Condition for 'moderately active':
        # FairlyActiveMinutes is greater than 20 or TotalSteps is between 7500 and 10000
        (df['FairlyActiveMinutes'] > 20) | ((df['TotalSteps'] > 7500) & (df['TotalSteps'] <= 10000)),
        
        # If neither of the above conditions are met, it's considered 'lightly active'
        True  # This condition will always be true, catching all remaining cases
    ]
    
    # Define the choices corresponding to each condition
    choices = ['very active', 'moderately active', 'lightly active']
    
    # Use numpy.select to create labels based on the conditions
    # If none of the conditions are met (which shouldn't happen due to the 'True' condition),
    # the default value 'lightly active' will be used
    y = np.select(conditions, choices, default='lightly active')
    
    # Encode the string labels into numerical values
    # This is necessary because most machine learning models require numerical inputs
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split the data into training and testing sets
    # 80% of the data will be used for training, and 20% for testing
    # random_state is set to ensure reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Return the split datasets and the LabelEncoder
    # The LabelEncoder is returned so it can be used later to convert
    # numerical predictions back into human-readable activity levels
    return (X_train, X_test, y_train, y_test), le

def visualize_data(X, y, label_encoder):
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue']
    
    #TODO
    # actually plot the data
        
    plt.xlabel('Total Steps')
    plt.ylabel('Very Active Minutes')
    plt.title('Activity Classification Visualization')
    plt.legend()
    plt.show()

    # PCA visualization
    #TODO
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Activity Classification Visualization (PCA)')
    plt.legend()
    plt.show()

    plt.savefig('activity_visualization.png')
    plt.close()

    plt.savefig('activity_visualization_pca.png')
    plt.close()
    

def train_model(train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = ActivityClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    best_val_acc = 0
    
    #TODO
    return model

def predict_activity(model, features, scaler, label_encoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    scaled_features = scaler.transform(features)
    with torch.no_grad():
        inputs = torch.FloatTensor(scaled_features).to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        return label_encoder.inverse_transform(predicted.cpu().numpy())


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
  # Prepare data
  (X_train, X_test, y_train, y_test), label_encoder = prepare_data(activity_df)

    # Visualize data
  visualize_data(X_train, y_train, label_encoder)

  # Scale features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Create data loaders
  train_dataset = ActivityDataset(X_train_scaled, y_train)
  val_dataset = ActivityDataset(X_test_scaled, y_test)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32)

  # Train model
  model = train_model(train_loader, val_loader)

  # Make predictions
  new_data = np.array([[10000, 8.0, 45]])  # steps, distance, active_time
  predicted_activity = predict_activity(model, new_data, scaler, label_encoder)
  print(f"Predicted activity level: {predicted_activity[0]}")
  