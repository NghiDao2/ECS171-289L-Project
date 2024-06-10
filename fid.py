import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

def get_features(model, image_tensors, transform, device='cuda'):
    processed_images = []
    # Process images for input
    for img in image_tensors:
        img = transform(img)  # Apply transformations
        processed_images.append(img)

    # Batch size
    batch_size = 50
    num_batches = len(processed_images) // batch_size + (1 if len(processed_images) % batch_size != 0 else 0)
    features_list = []

    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(processed_images))
            batch = torch.stack(processed_images[batch_start:batch_end]).to(device)
            features = model(batch)
            features_list.append(features.cpu().detach().numpy())

    # Concatenate all batch features
    return np.concatenate(features_list, axis=0)
def fid_score(generated_images, real_images, device='cuda'):
    
    # Load Inception model
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Removing the final fully connected layer
    model = model.to(device)
    model.eval()

    # Define transformation: resizing and normalization
    transform = transforms.Compose([
        transforms.Resize(299),  # Resize to 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to process and get features from image tensors

    # Calculate features
    fake_features = get_features(model, generated_images, transform, device=device)
    real_features = get_features(model, real_images, transform, device=device)

    # Calculate mean and covariance
    mu1, sigma1 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    mu2, sigma2 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid