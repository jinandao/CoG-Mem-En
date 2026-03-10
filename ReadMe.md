# CoG-Mem: Context-to-Memory Compression and Reasoning

This project focuses on memory enhancement for Large Language Models (LLMs), covering the entire pipeline from context memory compression and memory retrieval/filtering to memory-based dialogue generation. 

---

## 📂 Project Structure

```plaintext
CoG-Mem/
├── Configs/                # Experimental configurations for run_demo
├── Datasets/               # Datasets
├── Models/                 # Pre-trained/Open-source weights (for direct run_demo.sh execution)
├── Output/                 # Output path for model checkpoints and logs
├── Logs/                   # Execution logs (Training convergence, reasoning traces, and ablation results)
├── memory_compress_encoding_en.sh   # Core script: Executes Curriculum Learning (SFT -> DPO) for encoding
├── memory_compress_encoding_sft.py  # Implementation of SFT for memory encoding
├── memory_compress_encoding_dpo.py  # Implementation of DPO for memory encoding
├── memory_retrieval_sft_en.sh     # Script: Reproduces end-to-end memory retrieval experiments
├── memory_retrieval_sft.py        # Implementation of fine-tuning for memory filtering/retrieval
├── memory_synthesis_sft_en.sh # Script: Reproduces memory-based synthesis experiments
├── memory_synthesis_sft.py # Implementation of fine-tuning for memory-based dialogue
├── run_demo.sh             # Script: Demo for non-parametric learning performance
├── run_demo.py             # Entry point for the Demo
├── run_qwen3_32B_demo.sh   # Script: Demonstration of Qwen3-32B-FP8 Performance in Azeroth Experiments
└── run_qwen3_32B_demo.py   # Entry point for Azeroth Experiments using Qwen3-32B-FP8
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
### Download model weights
Download model weights from https://drive.google.com/drive/folders/1a92wBtBG9-oy3r9mTe8N2Dt0JuJ38SPf 
and place them in the corresponding paths within the ./Models/ directory.


## 🚀Training Pipeline and Reproduction
All training workflows are encapsulated in .sh scripts. Upon completion, test set outputs are automatically printed to facilitate direct performance evaluation.

**Hardware Note**: All experiments were conducted and reproduced on a **modified NVIDIA RTX 4090 with 48GB VRAM**. Users may need to adjust the `batch_size`, `max_length`, or file paths to fit their specific experimental environment and hardware constraints.

**1. Memory Encoding**, This experiment utilizes a Curriculum Learning strategy: it begins with Supervised Fine-Tuning (SFT) for structural alignment, followed by Direct Preference Optimization (DPO) to refine compression quality
```plaintext
   bash memory_compress_encoding_en.sh
```
Expected Result: Extracted "logical chunks" from compressed dialogues will be displayed. The test set includes manually curated "ground truth" logic chunks, allowing for a direct comparison with the model's output to evaluate performance.

**2. Memory Retrieval**，Reproduces the end-to-end filtering logic to verify the model’s ability to extract critical information from a vast memory pool.
```plaintext
   bash memory_retrieval_sft_en.sh
```
Expected Result: LLM filtering results compared against ground truth IDs will be printed, with the test accuracy provided at the end. Note: The test set uses newly generated data to address instances of memory reuse in the training set; meanwhile, the data for memory encoding and memory-based synthesis remain strictly independent and are free from this issue.

**3. Memory-based Synthesis Conversation**，Verifies the model's integrity in utilizing retrieved memories for downstream dialogue generation.
```plaintext
   bash memory_synthesis_sft_en.sh
```
Expected Result: AI-generated responses based on user input and retrieved context will be printed. The test set includes human-curated "Chain-of-Thought" (think) reasoning and corresponding ground-truth responses, enabling a detailed comparison to evaluate the model’s synthesis performance.

## 🧪Demo: Non-parametric Learning
A rapid demonstration script is provided to observe LLM behavior when encountering novel knowledge (e.g., modified formulas).
```plaintext
   bash run_demo.sh
```
Configuration: Internal paths for demo1_En and demo2_En contain JSON data with modified physical laws. Different counterfactual rules have also been configured for demo3_En to demo10_En, which can demonstrate the effect.
Features: Observe whether the LLM can understand and apply newly defined formulas (Logical Arbitration).

Note: While users can fine-tune formula data, absolute generalization accuracy is not guaranteed for custom inputs due to the small scale of the Demo dataset. The demo includes human-curated dialogues of physics calculations using Azeroth's rules, which will be printed sequentially—with ground-truth data appearing first, followed by the model's generated response—to facilitate performance comparison.

## 🧪Comparative experiment using Qwen3-32B-FP8
Requires significant VRAM (e.g., 2x 3090/4090 or A100)
```plaintext
   bash run_qwen3_32B_demo.sh
```

⚠️ Note on Dataset IDs: Portions of the data have been manually annotated and refined. Due to the focus on content quality during processing, some ID fields may appear non-sequential or logically inconsistent.
Please note: Since the training and evaluation logic (including compression and retrieval) does not rely on ID fields for indexing, this issue does not affect the validity of the dataset, model training, or the reproduction of experimental results.