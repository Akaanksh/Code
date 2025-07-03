# Transformer-based Machine Translation

This project implements a **Transformer architecture from scratch** in PyTorch for the task of bilingual machine translation using the [Hugging Face Datasets](https://huggingface.co/docs/datasets/) library. It includes a complete training and evaluation pipeline, tokenizer training, model building, and validation with common metrics.

## Overview

The Transformer model, introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), relies entirely on attention mechanisms and avoids recurrence. This implementation stays true to the original architecture, including:

- Multi-head self-attention
- Layer normalization and residual connections
- Positional encodings
- Encoder–decoder design
- Custom dataset and batching
- Greedy decoding for inference

The project supports training on any bilingual language pair (e.g., English–Hindi), and includes support for checkpointing, logging via TensorBoard, and BLEU/CER/WER evaluation.

---

## Project Structure
- model.py - Transformer model implementation
- train.py - Full training loop with validation
- dataset.py - Custom PyTorch Dataset for bilingual translation
- config.py - Hyperparameters and configuration
- weights/ - Checkpoints saved during training
- requirements.txt - Dependancies
- README.md - This file

---

## Features

- ✅ **Transformer built from scratch (no `nn.Transformer`)**
- ✅ Works with any HuggingFace `translation` dataset
- ✅ Word-level tokenizers trained per language using `tokenizers`
- ✅ Custom masking logic (padding and causal masks)
- ✅ Efficient greedy decoding for inference
- ✅ Evaluation: BLEU, Character Error Rate (CER), and Word Error Rate (WER)

---


## Model Details
- d_model: 512 (dimension of embeddings and attention layers)
- N: 6 encoder and decoder layers
- h: 8 attention heads
- d_ff: 2048 (hidden size of feed-forward layers)
- Label smoothing and dropout regularization included

---

##  Notes
- This implementation is educational, designed to closely mirror the original Transformer structure.
- Inference uses greedy decoding. Beam search can be added for better results.
- Larger datasets and BPE-based tokenization (like SentencePiece) can significantly improve performance.

---

## What I Learned
Building a Transformer from scratch taught me several subtle but important lessons:
- Attention replaces both RNNs and CNNs. It allows parallelism and better context modeling, which fundamentally changes how sequence tasks are approached.
- Masking is critical. Padding and causal masks must be handled correctly. If not, training behaves unpredictably or silently fails.
- Greedy decoding is limited. It works but often produces shorter or repetitive outputs. Beam search or other strategies are needed for better generation.
- Embedding size acts as a bottleneck. The d_model value controls how much information can be processed. Too small and learning is limited, too large and the model may become unstable.
- Tokenization influences everything. A poorly trained tokenizer leads to bad embeddings and sparse inputs. This impacts the entire pipeline.
- Training is sensitive. Tiny changes in learning rate, label smoothing, or batch size can drastically affect results.
