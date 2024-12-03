import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

#download dataset - https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz

#simple - preprocess data to fit size and convert to tensors(for pytorch)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



#referred to as backbone
class FeatureExtractor(nn.Module): 
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT) #start with default weights
        self.model.eval()
        
        for param in self.model.parameters():         # Freeze parameters(using resnet mainly for feature extraction)
            param.requires_grad = False
            
        def hook(module, input, output):
            self.features.append(output)
            
        self.model.layer3[-1].register_forward_hook(hook)# Register hook for layer3 only (no particular reason -- just performed the best)

    
    def forward(self, x):
        self.features = []
        with torch.no_grad(): #nograd because dont need backpropagation
            _ = self.model(x)
        return self.features[0]  #return features of that layer by forward pass




#workflow for autoenc-- resnet features from l3 - Input (1024) → 512 → 128 (bottleneck) → 512 → 1024 (output)

#referred to as model
class SimpleAutoencoder(nn.Module): 
    def __init__(self, in_channels=1024):  # Layer3 output channels are 1024
        super(SimpleAutoencoder, self).__init__()
        
        # Simplified encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),     #activation
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, in_channels, kernel_size=1)
        )

    def forward(self, x):    
        x = self.encoder(x)
        x = self.decoder(x)
        return x






def decision_function(segm_map):
    mean_top_10_values = []   #take the mean of the top 10 values of the highest recon error
    for map in segm_map:
        flattened = map.reshape(-1)      #map: 2d map -> 1d array
        sorted_values, _ = torch.sort(flattened, descending=True)   
        mean_top_10 = sorted_values[:10].mean()
        mean_top_10_values.append(mean_top_10)
    return torch.stack(mean_top_10_values)






def train_model(train_loader, test_loader, model, backbone, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    training_losses = []
    validation_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for data, _ in train_loader:
            data = data.cuda()
            with torch.no_grad():
                features = backbone(data)
            
            output = model(features)
            loss = criterion(output, features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # New weights = old weights - learning_rate * gradients
            
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.cuda()
                features = backbone(data)
                output = model(features)
                val_loss += criterion(output, features).item()
        
        training_losses.append(epoch_loss / len(train_loader))
        validation_losses.append(val_loss / len(test_loader))
        
        if epoch % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {training_losses[-1]:.4f}, Val Loss: {validation_losses[-1]:.4f}')
    
    return training_losses, validation_losses











def calculate_threshold(model, backbone, train_loader):
    model.eval()
    recon_errors = []
    
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.cuda()
            features = backbone(data)
            recon = model(features)
            
            segm_map = ((features - recon) ** 2).mean(axis=1)  #mse
            anomaly_score = decision_function(segm_map)        
            recon_errors.append(anomaly_score)
    
    recon_errors = torch.cat(recon_errors).cpu().numpy()
    threshold = np.mean(recon_errors) + 3 * np.std(recon_errors)    #covers distribution
    return threshold, recon_errors










# def predict_anomaly(image_path, model, backbone, threshold):
#     image = Image.open(image_path)
#     image = transform(image).unsqueeze(0).cuda()  #unsqueeze to add a dimension for 
    
#     with torch.no_grad():
#         features = backbone(image)
#         recon = model(features)
        
#     segm_map = ((features - recon) ** 2).mean(axis=1)
#     anomaly_score = decision_function(segm_map)
    
#     is_anomaly = anomaly_score >= threshold
#     return is_anomaly, anomaly_score, segm_map







def save_models(backbone, autoencoder, save_path='anomaly_models.pth'):
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'autoencoder_state_dict': autoencoder.state_dict(),
    }, save_path)

def load_models(model_path='anomaly_models.pth'):
    backbone = FeatureExtractor().cuda()
    model = SimpleAutoencoder().cuda()
    
    checkpoint = torch.load(model_path)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    
    return backbone, model








# def analyze_threshold_distribution(model, backbone, data_path):
#     """Analyze and visualize threshold distribution for OK and NOK images"""
#     model.eval()
#     backbone.eval()
    
#     ok_scores = []
#     nok_scores = []
    
#     # Process OK images (from train folder - all are OK)
#     train_dataset = ImageFolder(root=str(Path(data_path)/'train'), transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
#     with torch.no_grad():
#         for data, _ in train_loader:
#             data = data.cuda()
#             features = backbone(data)
#             recon = model(features)
#             segm_map = ((features - recon) ** 2).mean(axis=1)
#             scores = decision_function(segm_map)
#             ok_scores.extend(scores.cpu().numpy())
    
#     # Process NOK images (from test folder - except 'good' subfolder)
#     test_path = Path(data_path)/'test'
#     for defect_path in test_path.glob('*/*.png'):
#         if 'good' not in str(defect_path):
#             image = transform(Image.open(defect_path)).unsqueeze(0).cuda()
#             with torch.no_grad():
#                 features = backbone(image)
#                 recon = model(features)
#                 segm_map = ((features - recon) ** 2).mean(axis=1)
#                 score = decision_function(segm_map)
#                 nok_scores.extend(score.cpu().numpy())
    
#     ok_scores = np.array(ok_scores)
#     nok_scores = np.array(nok_scores)
    
#     # Calculate threshold
#     threshold = np.mean(ok_scores) + 3 * np.std(ok_scores)
    

    
#     return threshold

# Training and saving models
backbone = FeatureExtractor().cuda()
model = SimpleAutoencoder().cuda()

train_dataset = ImageFolder(root='leather/train', transform=transform)
train_data, test_data = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

losses = train_model(train_loader, test_loader, model, backbone)
save_models(backbone, model, 'anomaly_models.pth')

# Later: Loading models and analyzing threshold
backbone, model = load_models('anomaly_models.pth')
threshold = calculate_threshold(model, backbone, train_loader=train_loader)

# Make predictions using loaded models
#image_path = 'metal_nut/bent/000.png'
#is_anomaly, score, heatmap = predict_anomaly(image_path, model, backbone, threshold)
#print(f"Anomaly detected: {is_anomaly}")
#print(f"Anomaly score: {score.item():.4f}")
