#!/usr/bin/env python3

import logging
import os
import subprocess

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_directory_size(path):
    """Get directory size using du command"""
    try:
        # Use du command to get actual disk usage
        result = subprocess.run(
            ["du", "-sm", path], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            # du output format is "size_in_mb path"
            size_mb = float(result.stdout.split()[0])
            return size_mb
        return 0
    except Exception as e:
        logger.error(f"Error getting directory size: {str(e)}")
        return 0


def check_cache_locations():
    """Print cache locations and sizes"""
    # Check Hugging Face cache
    hf_cache = os.getenv(
        "TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface")
    )
    torch_cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")

    logger.info(f"Hugging Face cache directory: {hf_cache}")
    logger.info(f"PyTorch cache directory: {torch_cache}")

    # Get cache sizes if directories exist
    if os.path.exists(hf_cache):
        size_mb = get_directory_size(hf_cache)
        logger.info(f"Hugging Face cache size: {size_mb:.2f}MB")

    if os.path.exists(torch_cache):
        size_mb = get_directory_size(torch_cache)
        logger.info(f"PyTorch cache size: {size_mb:.2f}MB")


def download_models():
    """Explicitly download and cache models"""
    model_name = "openai/clip-vit-large-patch14"
    logger.info(f"Downloading and caching model: {model_name}")

    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        logger.info("Model and processor downloaded successfully")
        return model, processor
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        raise


def classify_image(image_path):
    # 1. Check for GPU and set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # 2. Load CLIP model and processor
    model, processor = download_models()
    model = model.to(device)
    model.eval()

    # 3. Load image
    image = Image.open(image_path)

    # 4. Define candidate labels
    candidate_labels = [
        "Hulk from Marvel Comics",
        "Bruce Banner",
        "Green monster",
        "Superhero",
        "Marvel character",
        "Comic book character",
        "Human figure",
        "Muscular character",
        "Action figure",
        "Movie character",
    ]

    # 5. Process inputs
    inputs = processor(
        images=image, text=candidate_labels, return_tensors="pt", padding=True
    )

    # 6. Move inputs to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 7. Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    # 8. Create sorted results
    results = list(zip(candidate_labels, probs.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    # 9. Print and store sorted results
    sorted_results = []
    for label, prob in results:
        result = f"{label}: {prob * 100:.1f}%"
        sorted_results.append(result)
        print(result)

    # 10. Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sorted_results


if __name__ == "__main__":
    try:
        # Check cache locations and sizes
        check_cache_locations()

        # Check CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info("GPU Memory Usage:")
            logger.info(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
            logger.info(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB")

        # Check MPS availability
        logger.info(f"MPS available: {torch.mps.is_available()}")

        image_path = "hulk.jpeg"
        results = classify_image(image_path)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
