import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch.nn.functional as F

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Model Definitions
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        def hook(module, input, output):
            self.features.append(output)
            
        self.model.layer3[-1].register_forward_hook(hook)
    
    def forward(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.model(x)
        return self.features[0]

class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels=1024):
        super(SimpleAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
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
    mean_top_10_values = []
    for map in segm_map:
        flattened = map.reshape(-1)
        sorted_values, _ = torch.sort(flattened, descending=True)
        mean_top_10 = sorted_values[:10].mean()
        mean_top_10_values.append(mean_top_10)
    return torch.stack(mean_top_10_values)

def load_models(model_path='anomaly_models.pth'):
    backbone = FeatureExtractor().cuda()
    model = SimpleAutoencoder().cuda()
    
    checkpoint = torch.load(model_path, weights_only=True)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    
    backbone.eval()
    model.eval()
    
    return backbone, model

def visualizations(model, backbone, data_path, threshold):
    """Generate comprehensive visualizations for anomaly detection analysis"""
    model.eval()
    backbone.eval()

    def plot_roc_curve(y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='red', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        return roc_auc

    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def analyze_threshold_distribution(model, backbone, data_path):
        ok_scores = []
        nok_scores = []
        
        train_dataset = ImageFolder(root=str(Path(data_path)/'train'), transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.cuda()
                features = backbone(data)
                recon = model(features)
                segm_map = ((features - recon) ** 2).mean(axis=1)
                scores = decision_function(segm_map)
                ok_scores.extend(scores.cpu().numpy())
        
        test_path = Path(data_path)/'test'
        for defect_path in test_path.glob('*/*.png'):
            if 'good' not in str(defect_path):
                image = transform(Image.open(defect_path)).unsqueeze(0).cuda()
                with torch.no_grad():
                    features = backbone(image)
                    recon = model(features)
                    segm_map = ((features - recon) ** 2).mean(axis=1)
                    score = decision_function(segm_map)
                    nok_scores.extend(score.cpu().numpy())
        
        ok_scores = np.array(ok_scores)
        nok_scores = np.array(nok_scores)
        threshold = np.mean(ok_scores) + 3 * np.std(ok_scores)
        
        plt.figure(figsize=(12, 6))
        plt.hist(ok_scores, bins=30, alpha=0.5, color='green', label='OK images')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
        plt.hist(nok_scores, bins=30, alpha=0.5, color='red', label='NOK images')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        false_positives = np.sum(ok_scores > threshold)
        true_positives = np.sum(nok_scores > threshold)
        false_negatives = np.sum(nok_scores <= threshold)
        true_negatives = np.sum(ok_scores <= threshold)
        
        accuracy = (true_positives + true_negatives) / (len(ok_scores) + len(nok_scores))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return threshold, ok_scores, nok_scores

    def plot_score_distribution_by_defect_type():
        scores_by_type = {}
        test_path = Path(data_path)/'test'
        
        for defect_type in test_path.glob('*'):
            scores = []
            for img_path in defect_type.glob('*.png'):
                img = transform(Image.open(img_path)).unsqueeze(0).cuda()
                with torch.no_grad():
                    features = backbone(img)
                    recon = model(features)
                    segm_map = ((features - recon) ** 2).mean(axis=1)
                    score = decision_function(segm_map)
                    scores.append(score.item())
            scores_by_type[defect_type.name] = scores
        
        plt.figure(figsize=(12, 6))
        plt.boxplot(scores_by_type.values(), labels=scores_by_type.keys())
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.xticks(rotation=45)
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Score Distribution by Defect Type')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_sample_predictions(num_samples=5):
        plt.figure(figsize=(15, 3*num_samples))
        test_path = Path(data_path)/'test'
        
        normal_paths = list((test_path/'good').glob('*.png'))
        anomaly_paths = []
        for defect_type in test_path.glob('*'):
            if defect_type.name != 'good':
                anomaly_paths.extend(list(defect_type.glob('*.png')))
        
        normal_samples = np.random.choice(normal_paths, num_samples)
        anomaly_samples = np.random.choice(anomaly_paths, num_samples)
        
        for idx, (normal_path, anomaly_path) in enumerate(zip(normal_samples, anomaly_samples)):
            normal_img = transform(Image.open(normal_path)).unsqueeze(0).cuda()
            with torch.no_grad():
                features = backbone(normal_img)
                recon = model(features)
                normal_map = ((features - recon) ** 2).mean(axis=1)
                normal_score = decision_function(normal_map)
            
            anomaly_img = transform(Image.open(anomaly_path)).unsqueeze(0).cuda()
            with torch.no_grad():
                features = backbone(anomaly_img)
                recon = model(features)
                anomaly_map = ((features - recon) ** 2).mean(axis=1)
                anomaly_score = decision_function(anomaly_map)
            
            plt.subplot(num_samples, 4, idx*4 + 1)
            plt.imshow(normal_img.squeeze().permute(1,2,0).cpu())
            plt.title(f'Normal\nScore: {normal_score.item():.4f}')
            plt.axis('off')
            
            plt.subplot(num_samples, 4, idx*4 + 2)
            plt.imshow(F.interpolate(normal_map.unsqueeze(1), size=(224, 224), 
                    mode='bilinear').squeeze().cpu(), cmap='jet')
            plt.title('Normal Heatmap')
            plt.axis('off')
            
            plt.subplot(num_samples, 4, idx*4 + 3)
            plt.imshow(anomaly_img.squeeze().permute(1,2,0).cpu())
            plt.title(f'Anomaly\nScore: {anomaly_score.item():.4f}')
            plt.axis('off')
            
            plt.subplot(num_samples, 4, idx*4 + 4)
            plt.imshow(F.interpolate(anomaly_map.unsqueeze(1), size=(224, 224), 
                    mode='bilinear').squeeze().cpu(), cmap='jet')
            plt.title('Anomaly Heatmap')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Collect data for ROC and confusion matrix
    y_true = []
    y_scores = []
    test_path = Path(data_path)/'test'
    
    for img_path in test_path.rglob('*.png'):
        is_anomaly = img_path.parent.name != 'good'
        img = transform(Image.open(img_path)).unsqueeze(0).cuda()
        
        with torch.no_grad():
            features = backbone(img)
            recon = model(features)
            segm_map = ((features - recon) ** 2).mean(axis=1)
            score = decision_function(segm_map)
        
        y_true.append(int(is_anomaly))
        y_scores.append(score.item())
    
    if threshold is None:
        threshold, _, _ = analyze_threshold_distribution(model, backbone, data_path)

    
    # Generate all visualizations
    visualize_sample_predictions()
    plot_score_distribution_by_defect_type()
    roc_auc = plot_roc_curve(y_true, y_scores)
    plot_confusion_matrix(y_true, np.array(y_scores) > threshold)
    
    return roc_auc, threshold

def main():
    backbone, model = load_models('anomaly_models.pth')
    roc_auc, threshold = visualizations(model, backbone, 'leather', None)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Threshold: {threshold:.4f}")

if __name__ == "__main__":
    main()