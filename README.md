# LLaMA-Factory RL Fine-tuning

This repository contains code for reinforcement learning (RL) fine-tuning of language models using the LLaMA-Factory framework. It implements three main RL approaches: DPO (Direct Preference Optimization), PPO (Proximal Policy Optimization), and RFT (Reinforcement Fine-Tuning).

## Repository Structure 
LLaMA-Factory/
├── src/
│ ├── RL_LLM_ft/ # RL fine-tuning implementations
│ │ ├── evaluate_and_compare.py # Evaluation script for comparing models
│ │ ├── math_problem_solver.py # Script for solving math problems with fine-tuned models
│ │ ├── vanilla/ # Base implementations of RL algorithms
│ │ │ ├── dpo_train.py # DPO training implementation
│ │ │ ├── ppo_train.py # PPO training implementation
│ │ │ ├── rft_train.py # RFT training implementation
│ │ │ └── visualize_training.py # Visualization utilities for training metrics
│ ├── llamafactory/ # Core LLaMA-Factory framework
│ │ ├── api/ # API server implementation
│ │ ├── chat/ # Chat interface implementation
│ │ ├── data/ # Data processing utilities
│ │ └── ...
├── evaluation_results/ # Results from model evaluations
│ └── 20250307_175414/ # Example evaluation results
│ └── evaluation_report.md # Detailed evaluation report


## Features

- **Multiple RL Fine-tuning Methods**: Implementations of DPO, PPO, and RFT for language model fine-tuning
- **Evaluation Framework**: Tools to compare different fine-tuning approaches on arithmetic tasks
- **Visualization Tools**: Utilities to visualize training progress and results
- **Math Problem Solver**: A utility to test fine-tuned models on mathematical reasoning tasks

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- TRL (Transformer Reinforcement Learning) library

### Installationgit clone https://github.com/yourusername/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .



### Training a Model with DPO
python src/RL_LLM_ft/vanilla/dpo_train.py


### Training a Model with PPO
python src/RL_LLM_ft/vanilla/ppo_train.py

## Evaluation Results
python src/RL_LLM_ft/evaluate_and_compare.py

The repository includes evaluation results comparing the performance of different fine-tuning approaches on arithmetic tasks. For example, in one evaluation:

- PLAIN Model Accuracy: 0.500
- DPO Model Accuracy: 0.410
- PPO Model Accuracy: 0.600

This shows that PPO fine-tuning provided a 20% improvement over the base model for arithmetic reasoning tasks.


## Using the Math Problem Solver


python src/RL_LLM_ft/math_problem_solver.py


## Visualizing Training Progress

python src/RL_LLM_ft/vanilla/visualize_training.py



This generates visualizations of training metrics such as loss and reward over time.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- This project builds on the LLaMA-Factory framework
- Implementations are inspired by research in reinforcement learning for language models
