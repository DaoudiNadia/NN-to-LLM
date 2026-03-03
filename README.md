# NN-to-LLM Migration for Tabular Classification

This repository hosts the artefacts accompanying the paper **"Neural Networks Teach LLMs on Tabular Data Tasks"**.  


## Overview
The paper proposes a knowledge migration approach that extracts insights from trained neural networks and provides them to LLMs via the prompt for tabular data prediction, preserving learned patterns when transitioning from NNs to LLMs.


## Datasets

| Dataset    | Task                        |
|------------|-----------------------------|
| Churn      | Customer churn prediction   |
| Adult      | Income above 50K prediction |
| Credit-G   | Credit risk classification  |
| Higgsmal      | Signal vs background classification        |
| NHANES     | Vitamin D deficiency prediction        |


## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Extract Neural Network Knowledge
For each dataset, run the corresponding extraction script:
```bash
python extract_nn_knowledge_churn.py
python extract_nn_knowledge_adult.py
python extract_nn_knowledge_creditg.py
python extract_nn_knowledge_higgsmall.py
python extract_nn_knowledge_nhanes.py
```
Each script trains a neural network on the dataset and prints:

-   Feature Importance
-   Feature Interactions
-   Representation Clustering

### Step 2: Build Prompts

The printed output from Step 1 is used to construct the prompts for each dataset.
The raw and guided prompts are available in `config.py`file.

### Step 3: Run LLM Inference

For each dataset, run the corresponding inference script:
```bash
python run_inference_gpt_churn.py
python run_inference_gpt_adult.py
python run_inference_gpt_creditg.py
python run_inference_gpt_higgsmall.py
python run_inference_gpt_nhanes.py  
```

Each script evaluates GPT-5 on the test set using two prompt strategies:

-   **Raw prompt**: no neural network knowledge provided
-   **Guided prompt**: neural network knowledge included into the prompt

Accuracy scores for both methods are printed and predictions are saved to the `predictions/` directory.
