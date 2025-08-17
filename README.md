# Text-to-Image Generation with Fine-Tuning

A comprehensive project for generating high-quality images from text descriptions using state-of-the-art diffusion models with fine-tuning capabilities and cloud deployment.

## 🚀 Features

- **Text-to-Image Generation**: Generate images from natural language descriptions
- **Fine-tuning**: Customize models on your own datasets using LoRA (Low-Rank Adaptation)
- **Cloud Deployment**: Ready for deployment on AWS, Google Cloud, Azure, and Hugging Face Spaces
- **Web Interface**: User-friendly web application for image generation
- **API Endpoints**: RESTful API for integration with other applications
- **Model Management**: Version control and model serving capabilities
- **Multiple Model Support**: SDXL, SD 2.1, and SD 1.5 compatibility

## 🏗️ Architecture

```
text-to-image-generation/
├── models/                 # Fine-tuned model files
├── data/                   # Training datasets
├── training/               # Fine-tuning scripts
├── inference/              # Model inference and serving
├── web_app/                # Streamlit web interface
├── utils/                  # Utility functions for data, models, and training
├── configs/                # Configuration files
├── .github/workflows/      # GitHub Actions for CI/CD
├── tests/                  # Unit tests
└── app.py                  # Gradio app for Hugging Face Spaces
```

## 🛠️ Technologies Used

- **Model**: Stable Diffusion XL (SDXL) with LoRA fine-tuning
- **Framework**: PyTorch, Diffusers, Transformers
- **Fine-tuning**: PEFT (Parameter-Efficient Fine-Tuning)
- **Web Framework**: FastAPI, Streamlit, Gradio
- **Cloud**: Docker, Kubernetes, Terraform, Hugging Face Spaces
- **Monitoring**: MLflow, Weights & Biases

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for training)
- Docker (for containerized deployment)
- Cloud account (AWS/GCP/Azure) or Hugging Face account

## 📊 Dataset Requirements for Fine-Tuning

### Dataset Structure
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── captions.txt
└── metadata.jsonl
```

### Data Formats

#### Option A: Captions File
```
image1.jpg|a beautiful landscape with mountains and lake, high quality, detailed
image2.png|a portrait of a woman with long hair, professional photography
```

#### Option B: JSONL Format (Recommended)
```json
{"file_name": "image1.jpg", "text": "a beautiful landscape with mountains and lake, high quality, detailed"}
{"file_name": "image2.png", "text": "a portrait of a woman with long hair, professional photography"}
```

### Dataset Specifications
- **Minimum Size**: 100-500 image-text pairs
- **Recommended Size**: 1,000-10,000 pairs
- **Optimal Size**: 5,000+ pairs for best results
- **Image Resolution**: 512x512 minimum, 1024x1024 preferred
- **Image Format**: JPG, PNG, or WebP
- **Text Quality**: Detailed, descriptive, consistent captions

### Creating Sample Dataset
```python
from utils.data_utils import create_sample_dataset

# Create a sample dataset with 100 images
create_sample_dataset("my_dataset", num_samples=100)
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/whoishmk/text-to-image-generation.git
cd text-to-image-generation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Base Model

```bash
# For SDXL (recommended)
python -c "from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')"

# For SD 2.1
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1')"

# For SD 1.5
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

### 4. Prepare Your Dataset

```bash
# Create sample dataset for testing
python -c "from utils.data_utils import create_sample_dataset; create_sample_dataset('data/train', num_samples=100)"

# Or organize your own images and captions in the data/ directory
```

### 5. Run Fine-tuning

```bash
python training/train_lora.py --config configs/training_config.yaml
```

### 6. Start Web Application

```bash
# Streamlit app
streamlit run web_app/app.py

# FastAPI server
python inference/model_server.py

# Gradio app (for Hugging Face Spaces)
python app.py
```

## 📚 Detailed Implementation

### Fine-tuning Process

1. **Data Preparation**: Organize your image-caption pairs
2. **Model Selection**: Choose base model (SDXL, SD 2.1, etc.)
3. **LoRA Configuration**: Set rank, alpha, and target modules
4. **Training**: Execute fine-tuning with your dataset
5. **Evaluation**: Assess model quality and performance
6. **Export**: Save fine-tuned model for deployment

### Training Configuration

```yaml
# configs/training_config.yaml
model:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  lora_rank: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "out_proj"]

training:
  learning_rate: 1e-4
  batch_size: 4
  num_epochs: 100
  gradient_accumulation_steps: 4
  save_steps: 500
  eval_steps: 100

data:
  train_data_dir: "data/train"
  validation_data_dir: "data/val"
  max_length: 77
  resolution: 1024
```

## ☁️ Cloud Deployment

### Hugging Face Spaces (Recommended)

1. **Create Space**: Set up a new Space on Hugging Face
2. **Upload Code**: Push your repository to the Space
3. **Configure**: Set hardware requirements (CPU/GPU)
4. **Deploy**: Automatic deployment via GitHub Actions

### AWS Deployment

1. **EC2 Setup**: Launch GPU instance
2. **Docker Deployment**: Use provided Dockerfile
3. **Load Balancer**: Configure ALB for traffic distribution
4. **Auto Scaling**: Set up ASG for dynamic scaling

### Google Cloud Deployment

1. **GKE Cluster**: Deploy to Kubernetes
2. **Cloud Run**: Serverless deployment option
3. **Cloud Storage**: Store models and datasets
4. **Cloud Monitoring**: Monitor performance metrics

### Azure Deployment

1. **AKS Cluster**: Azure Kubernetes Service
2. **Azure Container Instances**: Serverless containers
3. **Azure ML**: Managed ML platform
4. **Application Insights**: Monitoring and logging

## 🔧 Configuration

### Environment Variables

```bash
# .env
MODEL_PATH=/path/to/fine-tuned/model
DEVICE=cuda
BATCH_SIZE=4
MAX_LENGTH=77
RESOLUTION=1024
API_KEY=your_api_key
HF_TOKEN=your_huggingface_token
```

### Model Serving

```python
# inference/model_server.py
from diffusers import StableDiffusionXLPipeline
import torch

class TextToImageModel:
    def __init__(self, model_path):
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipeline.to("cuda")
    
    def generate(self, prompt, **kwargs):
        return self.pipeline(prompt, **kwargs)
```

## 📊 Performance Optimization

- **Model Quantization**: INT8/FP16 precision
- **Batch Processing**: Efficient batch inference
- **Caching**: Model and result caching
- **Load Balancing**: Distribute requests across instances
- **Auto-scaling**: Dynamic resource allocation
- **LoRA Efficiency**: Parameter-efficient fine-tuning

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/
```

## 📈 Monitoring and Logging

- **MLflow**: Experiment tracking and model versioning
- **Weights & Biases**: Training monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboard
- **ELK Stack**: Log aggregation and analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion models
- Hugging Face for Diffusers library and Spaces platform
- Microsoft for PEFT implementation
- Open source community for contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/whoishmk/text-to-image-generation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/whoishmk/text-to-image-generation/discussions)
- **Wiki**: [Project Wiki](https://github.com/whoishmk/text-to-image-generation/wiki)

## 🔄 Version History

- **v1.0.0**: Initial release with basic fine-tuning
- **v1.1.0**: Added cloud deployment support
- **v1.2.0**: Enhanced web interface and API
- **v2.0.0**: Major refactor with improved architecture
- **v2.1.0**: Added Hugging Face Spaces deployment and Gradio interface

---

**Note**: This project requires significant computational resources for training. Consider using cloud-based GPU instances for optimal performance. For deployment, Hugging Face Spaces provides an easy and cost-effective solution.
