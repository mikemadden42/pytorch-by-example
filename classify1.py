#!/usr/bin/env python3

import torch
from PIL import Image
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l


def classify_image(image_path):
    # 1. Check for GPU and set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Load the model and weights
    weights = EfficientNet_V2_L_Weights.DEFAULT
    model = efficientnet_v2_l(weights=weights)
    model = model.to(device)  # Move model to GPU
    model.eval()

    # 3. Get the transforms from the weights
    preprocess = weights.transforms()

    # 4. Load and preprocess the image
    img = Image.open(image_path)
    img_processed = preprocess(img)

    # 5. Add batch dimension and move to GPU
    batch = img_processed.unsqueeze(0).to(device)

    # 6. Get prediction
    with torch.no_grad():
        prediction = model(batch).squeeze(0)
        probability = torch.nn.functional.softmax(prediction, dim=0)

    # 7. Get class names and find top 5 predictions
    class_names = weights.meta["categories"]
    top5_prob, top5_catid = torch.topk(probability, 5)

    # 8. Print results
    for i in range(5):
        print(f"{class_names[top5_catid[i]]}: {top5_prob[i].item()*100:.1f}%")

    # 9. Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Example usage with GPU check
if __name__ == "__main__":
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
        print("GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")

    image_path = "hulk.jpeg"
    classify_image(image_path)
