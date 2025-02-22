# LLM-Nexus
LLM Nexus is a research hub for advancing large language models, featuring cutting-edge code, benchmarks, and insights. It fosters open collaboration to push the boundaries of NLP and AI innovation.

## DeepSeek V3 Models
The `DeepSeek-V3` directory contains models and scripts for inference and training using the DeepSeek V3 architecture. This includes tools for converting weights between different formats, such as FP8 to BF16.

### Directory Structure
- `DeepSeek-V3/inference/`: Contains scripts for running inference and converting model weights.
- `DeepSeek-V3/training/`: Contains scripts and configurations for training models.

### Key Files
- `DeepSeek-V3/inference/fp8_cast_bf16.py`: Script for converting FP8 weights to BF16.
- `DeepSeek-V3/inference/requirements.txt`: Lists the dependencies required for running inference scripts.

### Installation
To install the required dependencies, navigate to the `DeepSeek-V3/inference/` directory and run:
```sh
pip install -r requirements.txt