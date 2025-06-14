# Multiple-Choice-QA-Bias-Relief-with-PIF-Method

## Overview

Large Language Models (LLMs) show great promise for natural language processing in scientific domains, but they often suffer from selection bias—where the positioning of answer choices skews results—leading to unreliable performance in multiple-choice question answering (QA). This project presents an optimized LLM-based system that minimizes selection bias and improves accuracy on science-focused multiple-choice questions.

Our approach combines:
- **Supervised Fine-Tuning (SFT)**
- **Prompt Engineering**
- **Point-wise Intelligence Feedback (PIF)**
- **Retrieval-Augmented Generation (RAG)**

We demonstrate that this pipeline not only boosts accuracy but also significantly reduces bias, establishing a robust foundation for further domain-specific LLM optimization.

---

## Introduction

Despite LLMs' strong capabilities, they remain vulnerable to selection bias in multiple-choice tasks, often favoring options like "A" or "C" due to intrinsic model biases rather than content understanding. Previous work—such as that by Xue et al.—has inspired our use of Point-wise Intelligence Feedback (PIF) and data augmentation to combat this bias. Additionally, we leverage Retrieval-Augmented Generation (RAG) to enhance contextual understanding, supplying the model with more relevant background information.

---

## Methodology

### Model Selection & Baseline Evaluation

We evaluated several LLMs, including:
- LLaMA 3.2 3B Instruct
- LLaMA 3.2 3B Pretrained
- LLaMA 3.1 8B Pretrained
- Mistral-7B Instruct v0.3

**LLaMA 3.2 3B Pretrained** was selected for its superior accuracy.

### Prompt Engineering

We compared two prompt types:
- **Multiple-Choice Prompting (MCP):** Directly contrasts all answer options in a single forward pass, improving both accuracy and efficiency.
- **Cloze Prompting (CP):** Considers each answer option independently.

MCP improved model accuracy from 0.32 to 0.56 and reduced inference time on 200 questions from 18 minutes to 10 minutes on an A100 GPU.

### Point-wise Intelligence Feedback (PIF) & Reweighted Symbol-Content Binding (RSCB)

To address selection bias:
- **RSCB** integrates both option symbols and content into the loss function, emphasizing hard-to-classify samples using a focal loss approach.
- **PIF** introduces negative symbol-content pairs in fine-tuning, encouraging the model to distinguish correct from incorrect options without requiring human-annotated preferences.

### Retrieval-Augmented Generation (RAG)

Using `txtai` embeddings with a Wikipedia index, our RAG system:
- Constructs queries from both questions and answer options,
- Retrieves the top 3 most relevant text chunks,
- Integrates retrieved context into model prompts for improved contextual reasoning.

---

## Experiments

### Data & Prompt Example


{
  "prompt": "Answer the following question based on the {context} by selecting one of the options. Your response should be in the format {option}, where {option} is A, B, C, D, or E.",
  "context": "Retrieved information",
  "question": "What is the term used...",
  "expected_output_format": "{option}"
}

### Evaluation Metrics

- **Accuracy**
- **Mean Average Precision at 3 (MAP@3)**
- **Selection Bias (μ_bias):** Measured using answer-moving attack.

### Training & Optimization

- Fine-tuning with Unsloth API (LoRA, r=16) and Hugging Face Trainer API
- Batch size: 2, Grad Accumulation: 4, LR: 2e-5, AdamW 8bit, Weight Decay: 0.01, Max Steps: 100
- Mixed-precision (BF16), LoRA alpha: 16, Dropout: 0, No bias terms

### Results

| Model                   | Accuracy | MAP@3 | Selection Bias (μ_bias) |
|-------------------------|----------|-------|-------------------------|
| Initial (MCP)           | 0.56     | 0.67  | 19.88%                  |
| RSCB trained            | 0.925    | 0.954 | 14.40%                  |
| PIF trained             | 0.937    | 0.96  | 14.12%                  |
| RAG                     | 0.963    | 0.98  | 13.84%                  |

---

## Conclusion

Incorporating PIF, RSCB, and RAG techniques results in significant improvements in accuracy and robustness for science multiple-choice QA. The combination of RSCB and PIF is especially effective in reducing selection bias, while RAG further boosts contextual understanding. This methodology offers a promising foundation for domain-specific LLM optimization.

---
