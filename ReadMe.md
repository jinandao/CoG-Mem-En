# CoG-Mem: Context-to-Memory Compression and Reasoning

This project focuses on memory enhancement for Large Language Models (LLMs), covering the entire pipeline from context memory compression and memory retrieval/filtering to memory-based dialogue generation. 

---

## 📂 Project Structure

```plaintext
CoG-Mem/
├── Configs/                    # Experimental configurations for run_demo
├── Datasets/                   # Datasets for training and evaluation
├── Output/                     # Output path for model checkpoints and logs
├── memory_compress_sft_en.sh   # Core script: Executes training for memory compression
├── memory_compress_sft_en.py   # Implementation of SFT for memory compression
├── memory_query_sft_en.sh      # Script: Reproduces end-to-end memory retrieval experiments
├── memory_query_sft_en.py         # Implementation of fine-tuning for memory filtering/retrieval
├── memory_trigger_reasoning_conversations.sh # Script: Automated execution of atomic capability internalization and multi-round dialogue collaborative fine-tuning
├── memory_trigger_and_reasoning_sft_en.py # Implementation: SFT for memory triggering and constrained reasoning capabilities
├── memory_conversations_en.py # Implementation: Multi-round dialogue training with specific data retention for trigger/reasoning stability
├── run_demo.sh                 # Script: Demo for non-parametric learning performance
├── run_demo_en_basic_instruction.py     # Demo: Basic instruction following and knowledge application
├── run_demo_en_composite.py             # Demo: Complex scenarios including temporal matching and conflict arbitration
├── run_demo_en_zero_knowledge.py        # Demo: Zero-knowledge fallback and cognitive honesty
├── run_qwen_model.sh           # Script: Inference experiments using the base Qwen model
├── run_qwen_model.py           # Implementation: Qwen model inference for comparative analysis
├── run_demo_constrained_inference.sh   # Script: Experiments on pure constrained inference performance
└── run_demo_constrained_inference.py   # Implementation: Benchmarking constrained inference against baseline LLMs
```

## ⚙️Environment Setup and Installation

### Create a Conda environment with Python 3.10
conda create -n cogmem python=3.10 -y
### Activate the environment
conda activate cogmem
### Install PyTorch (GPU version)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
### Install core dependencies
pip install -r requirements.txt


## 🚀Training Pipeline and Reproduction
All training workflows are encapsulated in .sh scripts. Upon completion, test set outputs are automatically printed to facilitate direct performance evaluation.

**Hardware Note**: All experiments were conducted and reproduced on a **modified NVIDIA RTX 4090 with 48GB VRAM**. Users may need to adjust the `batch_size`, `max_length`, or file paths to fit their specific experimental environment and hardware constraints.

**1. Memory Compression**, this experiment focuses on condensing extensive dialogue history into high-density logical nodes. 
```plaintext
   bash memory_query_sft_en.sh
```
Expected Result: The model will output extracted "logical chunks" from compressed dialogues. Performance is evaluated by a direct comparison between the model's output and the manually curated "ground truth" logic chunks in the test set, ensuring both informational density and format integrity.

**2. Memory Queryl**，Reproduces the end-to-end filtering logic to verify the model’s ability to extract critical information from a vast memory pool.
```plaintext
   bash memory_retrieval_sft_en.sh
```
Expected Result: LLM filtering results compared against ground truth IDs will be printed, with the test accuracy provided at the end. 

**3. Memory-based Synthesis Conversation**，Verifies the model's integrity in utilizing retrieved memories for multi-round dialogue. It employs a Two-Phase Curriculum Learning strategy:
Phase 1 (Atomic Internalization): Single-turn SFT to internalize core "meta-capabilities" of memory triggering and constrained reasoning.
Phase 2 (Collaborative Orchestration): Multi-round dialogue SFT with a subset of Phase 1 data retained. This ensures the model achieves flexible triggering and accurate reasoning within structured, multi-turn conversation templates.
```plaintext
   bash memory_trigger_reasoning_conversations.sh
```
Expected Result: AI-generated responses based on retrieved context. The test set includes human-curated Chain-of-Thought (think) traces and ground-truth responses to evaluate synthesis performance and reasoning accuracy.

## 🧪Demo: Non-parametric Learning
A rapid demonstration script is provided to observe LLM behavior when encountering novel knowledge (e.g., counterfactual formulas or Azeroth-specific rules).
```plaintext
   bash run_demo.sh
```
#### Dataset Composition: The demo includes 60 basic tutorials for knowledge teaching/application, 30 zero-knowledge fallback scenarios, and 24 composite cases involving temporal matching and conflict arbitration.
#### Features: Evaluates the model's ability to prioritize non-parametric memory over pre-trained priors (Logical Arbitration) and its adherence to strict constraints.
#### Comparison: The execution log sequentially prints the Reference (Ground-Truth) output followed by the Model's Generated Response, allowing for a direct assessment of accuracy and consistency.


Note: While users can fine-tune formula data, absolute generalization accuracy is not guaranteed for custom inputs due to the small scale of the Demo dataset. 

## 🧪Comparative experiment using Qwen3-32B-FP8
Requires significant VRAM (e.g., 2x 3090/4090 or A100)
```plaintext
   bash run_qwen_model.sh
```

⚠️ Note on Dataset IDs: Portions of the data have been manually annotated and refined. Due to the focus on content quality during processing, some ID fields may appear non-sequential or logically inconsistent.
Please note: Since the training and evaluation logic (including compression and retrieval) does not rely on ID fields for indexing, this issue does not affect the validity of the dataset, model training, or the reproduction of experimental results. Furthermore, while some demonstration data in the Configs directory overlaps with the test set, there is no overlap with the training set. Certain memory entry IDs in the reference dialogues within the demonstration data may be inconsistent, as these IDs were randomly generated for display purposes during the demonstration.