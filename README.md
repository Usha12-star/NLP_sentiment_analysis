### A Comparative Study of BERT, RoBERTa, and DistilBERT against Classical Machine Learning Models

---

## Project Overview

This project addresses the challenge of automatically classifying mental health-related text into 7 categories using both classical machine learning baselines and outperforming transformer models. It is designed as a comparative study to evaluate model performance, computational efficiency, and generalization on real-world noisy data.

**Dataset:** [Mental Health Data — Hugging Face, Kaggle](https://huggingface.co/datasets/btwitssayan/sentiment-analysis-for-mental-health)(https://www.kaggle.com/datasets/ushapoudellamgade/mental-health-data)  
**Total Samples:** 52,680 | **Classes:** 7 | **Platform:** Kaggle (GPU: NVIDIA Tesla T4)

---

## Repository Structure

```
├── nlp-sentimental-analysis.ipynb   
├── README.md                       
└── outputs/
    ├── class_distribution.png
    ├── text_length_distribution.png
    ├── model_comparison.png
    ├── all_confusion_matrices.png
    ├── results_summary.json
    ├── classification_reports.txt
    ├── BERT_training_history.png
    ├── RoBERTa_training_history.png
    └── DistilBERT_training_history.png
```

---

## Class Labels

| Category             | Samples | %     |
|----------------------|---------|-------|
| Normal               | 16,342  | 31.0% |
| Depression           | 15,404  | 29.2% |
| Suicidal             | 10,652  | 20.2% |
| Anxiety              | 3,841   | 7.3%  |
| Bipolar              | 2,777   | 5.3%  |
| Stress               | 2,587   | 4.9%  |
| Personality Disorder | 1,077   | 2.0%  |

---

## Requirements

### Python Version
```
Python 3.12+
```

### Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install scikit-learn
pip install imbalanced-learn
pip install nltk
pip install pandas numpy matplotlib seaborn
```

Or install all at once:
```bash : pip install torch transformers scikit-learn imbalanced-learn nltk pandas numpy matplotlib seaborn ```

---

##  How to Run

### Option 1: Run on Kaggle (Recommended)
1. Upload the notebook `nlp-sentimental-analysis.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/code) or 'huggingface/btwitssayan/ sentiment-analysis-for-mental-health' to [Kaggle Notebooks](https://www.kaggle.com/code or any other python notebook/ editor)
2. Add the dataset: **Mental Health Data** by `ushapoudellamgade` or **sentiment-analysis-for-mental-health** by btwitssayan
3. Enable **GPU Accelerator** (T4 x2 recommended)
4. Click **Run All**.

### Option 2: Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```
2. Install dependencies (see above).
3. Download the dataset from Kaggle and place it at:
   ```
   data/data.csv
   ```
4. Update `DATA_PATH` in the config cell:
   ```python
   DATA_PATH = "data/data.csv"
   OUTPUT_DIR = "outputs"
   ```
5. Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook nlp-sentimental-analysis.ipynb
   ```

---

## Configuration

All hyperparameters are centralized in **Cell 2 (Config)**:

| Parameter           | Value    | Description                          |
|---------------------|----------|--------------------------------------|
| `MAX_LEN`           | 128      | Max token length for transformers    |
| `BATCH_SIZE`        | 32       | Training batch size                  |
| `EPOCHS`            | 10       | Number of training epochs            |
| `LR`                | 2e-5     | Learning rate (AdamW)                |
| `WEIGHT_DECAY`      | 0.01     | L2 regularization                    |
| `WARMUP_FRACTION`   | 0.10     | Warmup scheduler fraction            |
| `GRAD_CLIP`         | 1.0      | Gradient clipping norm               |
| `TEST_SIZE`         | 0.15     | Test split ratio (15%)               |
| `VAL_SIZE`          | 0.15     | Validation split ratio (15%)         |
| `TFIDF_MAX_FEATURES`| 50,000   | Max features for TF-IDF              |
| `SEED`              | 42       | Random seed for reproducibility      |

---

## Pipeline Steps

```
1. Data Loading & EDA
   └── Load CSV → drop nulls → visualize class/length distributions

2. Preprocessing
   └── Lowercase → remove URLs/mentions/special chars
   └── Lemmatization (WordNet) → stopword removal → LabelEncoding

3. Data Splitting (Stratified 70/15/15)
   └── Same test set used for ALL models (fair comparison)

4. Class Imbalance Handling
   └── RandomOverSampler (for classical models only)

5. Classical Baseline Models (TF-IDF features)
   ├── Naive Bayes (MultinomialNB, α=0.1)
   ├── Logistic Regression (C=1.0, balanced)
   └── Linear SVC (C=1.0, balanced)

6. Transformer Models (raw text, class-weighted loss)
   ├── BERT (bert-base-uncased, 110M params)
   ├── RoBERTa (roberta-base, 125M params)
   └── DistilBERT (distilbert-base-uncased, 66M params)

7. Evaluation & Visualization
   └── Accuracy, Weighted F1, Confusion Matrices, Training Curves
```

---

## Results

| Model               | Accuracy | F1 (Weighted) | Params |
|---------------------|----------|---------------|--------|
| Naive Bayes         | 0.6369   | 0.6430        | —      |
| Logistic Regression | 0.7488   | 0.7473        | —      |
| Linear SVC          | 0.7346   | 0.7312        | —      |
| DistilBERT          | 0.8188   | 0.8192        | 66M    |
| BERT                | 0.8294   | 0.8299        | 110M   |
| **RoBERTa**         | **0.8366** | **0.8377**  | 125M   |

> **RoBERTa** achieves the best performance with **83.66% accuracy** and **0.8377 weighted F1**, outperforming classical models by ~10%.

---

## Handling Real-World Constraints

| Constraint              | Strategy Applied                                      |
|-------------------------|-------------------------------------------------------|
| Imbalanced dataset      | RandomOverSampler + class-weighted loss functions     |
| Noisy/unstructured text | Regex cleaning + lemmatization + stopword removal     |
| Computational limits    | GPU-first with automatic CPU fallback on OOM          |
| Generalization          | Stratified splits + warm-up scheduler + gradient clip |

---

## Output Files

After running the notebook, the following files are saved to `/kaggle/working/outputs/` (or your configured `OUTPUT_DIR`):

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart of class sample counts |
| `text_length_distribution.png` | Word count histogram per class |
| `model_comparison.png` | Accuracy & F1 comparison bar chart |
| `all_confusion_matrices.png` | Combined 2×3 grid of all 6 confusion matrices |
| `{model}_confusion_matrix.png` | Individual confusion matrix per model |
| `{model}_training_history.png` | Loss & accuracy curves (transformers only) |
| `results_summary.json` | JSON summary of all model scores |
| `classification_reports.txt` | Full per-class precision/recall/F1 reports |

---

## Notes

- Transformer models are downloaded automatically from [Hugging Face Hub](https://huggingface.co/) on first run.
- A free HuggingFace token (`HF_TOKEN`) can be set as a Kaggle secret to avoid rate-limit warnings.
- Total training time on Tesla T4 GPU: ~5.8 hours (all 6 models, 10 epochs each).
- CPU fallback is automatically triggered if CUDA runs out of memory.

---

## Author

**Usha Poudel Lamgade* — *MAS Data Science second semester project for natural language procession course*  
Dataset credit: [btwitssayan] (https://huggingface.co/datasets/btwitssayan/sentiment-analysis-for-mental-health) on hugging face
                                                OR
                [ushapoudellamgade](https://www.kaggle.com/datasets/ushapoudellamgade/mental-health-data) on Kaggle
                                           

