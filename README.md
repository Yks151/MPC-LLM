# Multimodal Physical-information Constraints LLM for Fault Diagnosis of Pressurized Water Reactor Coolant System

> **A Robust and Physically-Informed Large Language Model for Imbalanced and Limited Data Scenarios in Nuclear Power Plant Safety**

This repository contains the official implementation of the paper **"Multimodal Physical-information Constraints LLM for Fault Diagnosis of Pressurized Water Reactor Coolant System in Under-sampled and Imbalanced Data"**, presenting a novel fusion of deep learning with domain-specific physical laws to achieve reliable fault diagnosis in safety-critical nuclear systems under challenging data conditions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/xxxx.xxxxx)

---

## 🔬 Key Features

- **Multimodal Physical Constraints**: Integrates first-principle laws (mass, energy, momentum, neutron conservation) as soft constraints into LLM training.
- **Imbalance-Robust Architecture**: Designed specifically for highly imbalanced datasets with minority fault classes (e.g., LOCA).
- **Few-Shot Learning Capability**: Achieves high diagnostic accuracy with limited training samples per class.
- **Explainable AI Outputs**: Generates human-readable, physics-based explanations for each diagnosis.
- **Confidence Calibration**: Produces well-calibrated confidence scores suitable for safety-critical decision-making.

## 📊 Performance Highlights

| Metric | Our Model (MPI‑LLM) | Best Baseline |
|--------|----------------------|---------------|
| **Accuracy** | **92.5%** | 89.2% (Transformer‑TS) |
| **F1‑Score** | **0.910** | 0.871 (Inception Time) |
| **AUC‑ROC** | **0.942** | 0.917 (Transformer‑TS) |
| **Minority Class Accuracy** | **+31.6%** | Baseline at 1:10 imbalance |
| **False Alarm Reduction** | **67.3%** | Compared to baseline |
| **Early Detection Time** | **54.2% faster** | Over conventional methods |

## 📁 Repository Structure
