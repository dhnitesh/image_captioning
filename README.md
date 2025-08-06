# Image Captioning Project with ViT-GPT2

This project implements an image captioning system using the pre-trained **nlpconnect/vit-gpt2-image-captioning** model from Hugging Face. The model combines Vision Transformer (ViT) for image encoding and GPT-2 for text generation.

## Features

- 🖼️ **Pre-trained Model**: Uses the high-quality ViT-GPT2 model for immediate caption generation
- 🔧 **Fine-tuning Support**: Option to fine-tune the model on your custom dataset
- 📁 **Batch Processing**: Process multiple images at once
- 🎯 **Multiple Caption Generation**: Generate diverse captions for the same image
- 📊 **Detailed Reports**: Generate comprehensive reports of captioning results

## Project Structure

```
image_captioning/
├── data/
│   ├── train_images/      # Directory for training images
│   └── captions.json      # Annotations and captions for images
├── scripts/
│   ├── infer.py           # Single image inference script
│   ├── batch_process.py   # Batch processing script
│   └── train.py           # Fine-tuning script (optional)
├── models/
│   └── vit_gpt2_model.py  # ViT-GPT2 model wrapper
├── utils/
│   └── image_utils.py     # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. **Clone or download** this project to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** by running a test inference:
   ```bash
   cd scripts
   python infer.py --image path/to/test/image.jpg
   ```

## Usage

### Single Image Captioning

Generate a caption for a single image:

```bash
cd scripts
python infer.py --image path/to/your/image.jpg
```

**Advanced options**:
```bash
# Generate multiple diverse captions
python infer.py --image image.jpg --multiple 3 --temperature 1.2

# Customize caption length and beam search
python infer.py --image image.jpg --max_length 20 --num_beams 6

# Use a locally saved model
python infer.py --image image.jpg --model "../models/my_fine_tuned_model"
```

### Batch Processing

Process multiple images in a directory:

```bash
cd scripts
python batch_process.py --input path/to/image/directory --output captions.json
```

**Options**:
```bash
# Process recursively and generate report
python batch_process.py --input images/ --output results.json --report report.txt --recursive

# Custom model and parameters
python batch_process.py --input images/ --model custom_model --max_length 20 --num_beams 6
```

### Fine-tuning (Optional)

Fine-tune the model on your custom dataset:

1. **Prepare your data**:
   - Place training images in `data/train_images/`
   - Create `data/captions.json` with the format:
     ```json
     [
       {"image": "image1.jpg", "caption": "A description of image1"},
       {"image": "image2.jpg", "caption": "A description of image2"}
     ]
     ```

2. **Start fine-tuning**:
   ```bash
   cd scripts
   python train.py --epochs 5 --batch_size 4 --learning_rate 5e-5
   ```

## Model Information

- **Base Model**: `nlpconnect/vit-gpt2-image-captioning`
- **Vision Encoder**: Vision Transformer (ViT-base-patch16-224)
- **Text Decoder**: GPT-2
- **Input Resolution**: 224x224 pixels
- **Supported Formats**: JPG, JPEG, PNG, BMP, TIFF, WebP

## Examples

### Basic Usage
```python
from models.vit_gpt2_model import ViTGPT2ImageCaptioning

# Load the model
model = ViTGPT2ImageCaptioning()

# Generate a caption
caption = model.generate_caption("image.jpg")
print(caption)  # Output: "a dog sitting on a couch"

# Generate multiple captions
captions = model.generate_multiple_captions("image.jpg", num_captions=3)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_length` | Maximum caption length | 16 |
| `num_beams` | Beam search width | 4 |
| `temperature` | Sampling temperature | 1.0 |
| `multiple` | Number of captions to generate | 1 |

## Requirements

- Python 3.7+
- PyTorch ≥ 1.9.0
- Transformers ≥ 4.20.0
- Pillow (PIL)
- NumPy

## Troubleshooting

**GPU Memory Issues**: If you encounter CUDA out of memory errors, try:
- Reducing batch size: `--batch_size 2`
- Using CPU: The model will automatically fallback to CPU if CUDA is unavailable

**Model Loading Issues**: Ensure you have a stable internet connection for the first run to download the pre-trained model.

**Image Format Issues**: Ensure your images are in supported formats and not corrupted.

## License

This project uses the MIT License. The pre-trained model follows the license terms of the original nlpconnect/vit-gpt2-image-captioning model.

## License

[Your license information here]
