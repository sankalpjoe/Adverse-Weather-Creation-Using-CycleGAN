# Day-Night Image Translation with CycleGAN

This repository contains an implementation of CycleGAN for translating images between day and night domains. The model can transform daytime scenes to nighttime and vice versa while preserving the content and structure of the original images.

## Adverse Weather Creation using Unpaired Image-to-Image Translation

### Abstract

This project aims to develop an unsupervised image-to-image translation model that can effectively translate images from adverse weather conditions to standard conditions, enhancing the performance of autonomous systems in various environments. The proposed solution employs generative adversarial networks (GANs) to generate realistic low-visibility nighttime scenarios from high-visibility daytime datasets.

The key challenges addressed in this work include:
- Lack of availability of precisely aligned paired datasets
- Maintaining semantic consistency during translation
- Balancing the trade-off between generating diverse synthetic images and preserving the original content and structure

The project leverages the Berkeley DeepDrive (BDD100K) dataset for autonomous vehicles and the Landing Approach Runway Detection (LARD) dataset for aviation applications.

## Overview & Motivation

CycleGAN is an unsupervised image-to-image translation technique that doesn't require paired training data. This implementation focuses specifically on the day-night translation task, which can be useful for:

- Data augmentation for computer vision tasks
- Artistic rendering of landscapes
- Simulation of different lighting conditions
- Enhancing autonomous vehicle and aviation system performance in adverse conditions

The motivation behind this project stems from the critical need for robust image translation techniques that can operate effectively in real-world scenarios where obtaining paired training data is impractical or prohibitively expensive. By leveraging unpaired image-to-image translation methods such as CycleGAN, the project seeks to overcome these challenges and enable the generation of realistic images across diverse domains.

Image-to-image translation is a fundamental task in computer vision, with applications ranging from style transfer to semantic segmentation. Traditional methods rely on paired datasets, where each input image is associated with a corresponding output image, for training. However, acquiring such paired data can be difficult, if not impossible, in many real-world scenarios. Unpaired image-to-image translation techniques have emerged as a solution to this challenge, enabling the transformation of images between different domains without the need for paired examples.

## Project Structure

- `config.py`: Configuration settings for model training
- `dataset.py`: Custom dataset class for loading and transforming day/night images
- `discriminator_model.py`: Implementation of the discriminator neural network
- `generator_model.py`: Implementation of the generator neural network
- `train.py`: Main training script
- `utils.py`: Utility functions for saving/loading checkpoints and seeding

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Albumentations
- NumPy
- tqdm
- Pillow

## Setup

1. Clone this repository
2. Prepare your dataset with the following structure:
   ```
   data/
   ├── train/
   │   ├── days/
   │   │   ├── day_img_1.jpg
   │   │   ├── day_img_2.jpg
   │   │   └── ...
   │   └── nights/
   │       ├── night_img_1.jpg
   │       ├── night_img_2.jpg
   │       └── ...
   └── val/
       ├── days/
       │   ├── day_val_1.jpg
       │   ├── day_val_2.jpg
       │   └── ...
       └── nights/
           ├── night_val_1.jpg
           ├── night_val_2.jpg
           └── ...
   ```
3. Create a `saved_images` directory to store generated samples during training

## Training

To train the model with default settings:

```bash
python train.py
```

You can modify various hyperparameters in `config.py` including:
- Learning rate
- Batch size
- Number of epochs
- Lambda coefficients for identity and cycle consistency losses

## Implementation Details

This implementation includes several key components of CycleGAN:

### Generators
- Uses a ResNet-based architecture with 9 residual blocks
- Downsampling and upsampling layers for efficient processing
- Tanh activation in the final layer

### Discriminators
- PatchGAN discriminator that classifies patches as real or fake
- Enables the model to focus on texture and style

### Loss Functions
- Adversarial loss: Encourages generators to produce realistic images
- Cycle consistency loss: Ensures that translating an image to the target domain and back produces the original image
- Identity loss: Helps preserve color and content when appropriate

### Model Description in Detail

The Generative Adversarial Network (GAN) architecture is utilized for its versatility and remarkable outcomes across various applications, such as text-to-image and image-to-image translation. CycleGAN, a specific type of GAN designed for unpaired image-to-image translation, involves training two generator models and two discriminator models simultaneously:

- The discriminator (D) distinguishes between real and fake images
- The generator (G) learns the data distribution, setting the two neural networks in opposition

Unpaired image-to-image translation is crucial in computer vision and machine learning, enabling the transformation of images from one domain to another without needing paired examples in the training dataset. Traditional methods require paired datasets, which are often difficult to obtain in real-world scenarios, especially for:
- Extreme weather conditions
- Complex scenes
- Low visibility
- Night-time settings

CycleGAN addresses this challenge by learning mappings between images from different domains, such as transitions between clear and foggy weather, daylight and night-time scenes, or varying visibility conditions.

### Training Optimizations
- Gradient accumulation to handle memory constraints
- Mixed precision training with CUDA amp
- Periodic checkpoint saving

## Inference

After training, you can use the trained generators for inference:

```python
from generator_model import Generator
import torch
from PIL import Image
import torchvision.transforms as transforms
import config

# Load the trained generator
gen_H = Generator(img_channels=3, num_residuals=9)
gen_Z = Generator(img_channels=3, num_residuals=9)

checkpoint_H = torch.load("genh.pth.tar", map_location=config.DEVICE)
checkpoint_Z = torch.load("genz.pth.tar", map_location=config.DEVICE)

gen_H.load_state_dict(checkpoint_H["state_dict"])
gen_Z.load_state_dict(checkpoint_Z["state_dict"])

# Set to evaluation mode
gen_H.eval()
gen_Z.eval()

# Load and transform an image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# For day to night conversion
day_img = Image.open("path/to/day/image.jpg").convert("RGB")
day_tensor = transform(day_img).unsqueeze(0).to(config.DEVICE)
with torch.no_grad():
    night_tensor = gen_Z(day_tensor)

# Convert back to image and save
night_img = (night_tensor.squeeze(0) * 0.5 + 0.5).cpu().permute(1, 2, 0).numpy()
night_img = (night_img * 255).astype('uint8')
Image.fromarray(night_img).save("night_output.jpg")
```

## Memory Management

This implementation includes several optimizations for managing GPU memory:

- CUDA memory fraction setting
- Mixed precision training
- Gradient accumulation
- Explicit cache clearing

You may need to adjust these settings based on your hardware capabilities.

## Results


<img src="https://github.com/user-attachments/assets/bd6f56f1-11d4-4e14-a9bc-d22d094749f0" width="500" alt="image">

After training on the Berkeley DeepDrive dataset with 506 images for 35 epochs, the model demonstrated success in mapping images between clear and adverse conditions. The results showed promising transformations of:
- Daytime scenes to nighttime
- Clear driving conditions to rainy weather

The model's performance was sensitive to hyperparameters, suggesting that further optimization could improve results.

## Related Work

Various approaches have been developed to tackle computer vision challenges similar to those addressed in this project:

- **Unpaired image-to-image translation techniques**:
  - CycleGAN [7]: Converts images between domains without paired training data but lacks strong disentanglement abilities
  - UNIT [8]: Introduces shared latent spaces for better disentanglement
  - MUNIT and DRIT: Further decompose images into domain-invariant content and domain-specific styles
  - StarGAN: Enhances diversity through multi-domain translation

- **Low-light image enhancement**:
  - EnlightenGAN: Improves luminosity without paired data

- **Adverse weather vision tasks**:
  - ToDayGAN and Porav et al.'s methods: Improve image quality for localization and semantic segmentation tasks

## Future Directions

- Further advancements in disentanglement techniques for more robust and accurate image translations
- Exploration of comprehensive and domain-specific image enhancement techniques tailored to address unique challenges posed by adverse weather conditions
- Investigation of alternative methods such as paired image-to-image translation techniques and other adversarial learning approaches
- Incorporation of uncertainty-aware learning to address challenges in image translation under adverse weather conditions
- Scaling up the solution to handle larger datasets and integrating it with other techniques

## Acknowledgements

This implementation is based on the following research:
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros
- SPA-GAN: Spatial Attention GAN for Image-to-Image Translation (Emami et al., 2021)
- SuperstarGAN: Generative adversarial networks for image-to-image translation in large-scale domains (Ko et al., 2023)
- https://github.com/junyanz/CycleGAN

