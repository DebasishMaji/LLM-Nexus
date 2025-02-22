## Introduction

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 
To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. 
Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. 
We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. 
Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models.
Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training.
In addition, its training process is remarkably stable. 
Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks.

## Model Summary

---

**Architecture: Innovative Load Balancing Strategy and Training Objective**

- On top of the efficient architecture of DeepSeek-V2, we pioneer an auxiliary-loss-free strategy for load balancing, which minimizes the performance degradation that arises from encouraging load balancing.
-  We investigate a Multi-Token Prediction (MTP) objective and prove it beneficial to model performance. 
    It can also be used for speculative decoding for inference acceleration. 

---

**Pre-Training: Towards Ultimate Training Efficiency**

- We design an FP8 mixed precision training framework and, for the first time, validate the feasibility and effectiveness of FP8 training on an extremely large-scale model.  
- Through co-design of algorithms, frameworks, and hardware, we overcome the communication bottleneck in cross-node MoE training, nearly achieving full computation-communication overlap.  
  This significantly enhances our training efficiency and reduces the training costs, enabling us to further scale up the model size without additional overhead.  
- At an economical cost of only 2.664M H800 GPU hours, we complete the pre-training of DeepSeek-V3 on 14.8T tokens, producing the currently strongest open-source base model. The subsequent training stages after pre-training require only 0.1M GPU hours.

---

**Post-Training: Knowledge Distillation from DeepSeek-R1**

-   We introduce an innovative methodology to distill reasoning capabilities from the long-Chain-of-Thought (CoT) model, specifically from one of the DeepSeek R1 series models, into standard LLMs, particularly DeepSeek-V3. Our pipeline elegantly incorporates the verification and reflection patterns of R1 into DeepSeek-V3 and notably improves its reasoning performance. Meanwhile, we also maintain a control over the output style and length of DeepSeek-V3.

---

## How to Run Locally

DeepSeek-V3 can be deployed locally using the following hardware and open-source community software:

1. **DeepSeek-Infer Demo**: We provide a simple and lightweight demo for FP8 and BF16 inference.
2. **SGLang**: Fully support the DeepSeek-V3 model in both BF16 and FP8 inference modes, with Multi-Token Prediction [coming soon](https://github.com/sgl-project/sglang/issues/2591).
3. **LMDeploy**: Enables efficient FP8 and BF16 inference for local and cloud deployment.
4. **TensorRT-LLM**: Currently supports BF16 inference and INT4/8 quantization, with FP8 support coming soon.
5. **vLLM**: Support DeepSeek-V3 model with FP8 and BF16 modes for tensor parallelism and pipeline parallelism.
6. **AMD GPU**: Enables running the DeepSeek-V3 model on AMD GPUs via SGLang in both BF16 and FP8 modes.
7. **Huawei Ascend NPU**: Supports running DeepSeek-V3 on Huawei Ascend devices.

Since FP8 training is natively adopted in our framework, we only provide FP8 weights. If you require BF16 weights for experimentation, you can use the provided conversion script to perform the transformation.

Here is an example of converting FP8 weights to BF16:


```shell
cd inference
python3 -m venv venv
pip install -r requirements.txt

python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights
```

> [!NOTE]
> Hugging Face's Transformers has not been directly supported yet.

### Inference with DeepSeek-Infer Demo (example only)

#### System Requirements

> [!NOTE] 
> Linux with Python 3.10 only. Mac and Windows are not supported.

Dependencies:
```pip-requirements
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```
#### Model Weights & Demo Code Preparation

First, clone our DeepSeek-V3 GitHub repository:

```shell
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
```

Navigate to the `inference` folder and install dependencies listed in `requirements.txt`. Easiest way is to use a package manager like `conda` or `uv` to create a new virtual environment and install the dependencies.

```shell
cd DeepSeek-V3/inference
pip install -r requirements.txt
```

Download the model weights from Hugging Face, and put them into `/path/to/DeepSeek-V3` folder.

#### Model Weights Conversion

Convert Hugging Face model weights to a specific format:

```shell
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```

#### Run

Then you can chat with DeepSeek-V3:

```shell
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
```

Or batch inference on a given file:

```shell
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --input-file $FILE
```

### Inference with SGLang (recommended)

[SGLang](https://github.com/sgl-project/sglang) currently supports [MLA optimizations](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations), [DP Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models), FP8 (W8A8), FP8 KV Cache, and Torch Compile, delivering state-of-the-art latency and throughput performance among open-source frameworks.

Notably, [SGLang v0.4.1](https://github.com/sgl-project/sglang/releases/tag/v0.4.1) fully supports running DeepSeek-V3 on both **NVIDIA and AMD GPUs**, making it a highly versatile and robust solution.

SGLang also supports [multi-node tensor parallelism](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208), enabling you to run this model on multiple network-connected machines.

Multi-Token Prediction (MTP) is in development, and progress can be tracked in the [optimization plan](https://github.com/sgl-project/sglang/issues/2591).

Here are the launch instructions from the SGLang team: https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

### Inference with LMDeploy (recommended)
[LMDeploy](https://github.com/InternLM/lmdeploy), a flexible and high-performance inference and serving framework tailored for large language models, now supports DeepSeek-V3. It offers both offline pipeline processing and online deployment capabilities, seamlessly integrating with PyTorch-based workflows.

For comprehensive step-by-step instructions on running DeepSeek-V3 with LMDeploy, please refer to here: https://github.com/InternLM/lmdeploy/issues/2960


### Inference with TRT-LLM (recommended)

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) now supports the DeepSeek-V3 model, offering precision options such as BF16 and INT4/INT8 weight-only. Support for FP8 is currently in progress and will be released soon. You can access the custom branch of TRTLLM specifically for DeepSeek-V3 support through the following link to experience the new features directly: https://github.com/NVIDIA/TensorRT-LLM/tree/deepseek/examples/deepseek_v3. 


### Inference with vLLM (recommended)

[vLLM](https://github.com/vllm-project/vllm) v0.6.6 supports DeepSeek-V3 inference for FP8 and BF16 modes on both NVIDIA and AMD GPUs. Aside from standard techniques, vLLM offers _pipeline parallelism_ allowing you to run this model on multiple machines connected by networks. For detailed guidance, please refer to the [vLLM instructions](https://docs.vllm.ai/en/latest/serving/distributed_serving.html). Please feel free to follow [the enhancement plan](https://github.com/vllm-project/vllm/issues/11539) as well.

### Recommended Inference Functionality with AMD GPUs

In collaboration with the AMD team, we have achieved Day-One support for AMD GPUs using SGLang, with full compatibility for both FP8 and BF16 precision. For detailed guidance, please refer to the [SGLang instructions](#63-inference-with-lmdeploy-recommended).
