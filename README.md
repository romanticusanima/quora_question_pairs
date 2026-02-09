# Quora Question Pairs - Duplicate Detection

A comprehensive machine learning project for detecting duplicate questions on Quora using both traditional ML algorithms and modern transformer-based deep learning models.

## Overview

This project implements an end-to-end NLP pipeline to classify whether two questions are semantically equivalent (duplicates). The project compares multiple approaches including feature-engineered traditional ML models, TF-IDF-enhanced models, and transformer models.

## Project Structure

```
.
├── 01_data_EDA.ipynb          # Exploratory Data Analysis
├── 02_preprocessing.ipynb     # Data preprocessing & feature engineering
├── 03_models.ipynb            # Traditional ML models (LR, RF, XGB, LGBM)
├── 04_distil_bert.ipynb       # DistilBERT transformer model
├── data/                      # Dataset directory
│   ├── quora_question_pairs_train.csv.zip
│   ├── quora_question_pairs_test.csv.zip
│   └── processed_data/        # Preprocessed data with engineered features
├── model/                     # DistilBERT model configuration & tokenizer
└── results/                   # Serialized model evaluation metrics
```

## Dataset

**Source**: [Quora Question Pairs dataset](https://www.kaggle.com/competitions/quora-question-pairs)

**Statistics**:
- Training set: ~323,432 question pairs
- Test set: ~80,858 question pairs
- Target distribution: 37% duplicates, 63% non-duplicates

**Schema**:
- `id`: unique identifier for question pair
- `qid1`, `qid2`: unique ids of each question
- `question1`, `question2`: the full text of each question
- `is_duplicate`: target variable (1 = duplicate, 0 = not duplicate)

## Model Performance

Evaluation metrics: Log Loss (primary), F1-score

| Model | Log Loss (Train) | Log Loss (Val) | F1 Score (Train) | F1 Score (Val) |
|-------|------------------|----------------|------------------|----------------|
| **DistilBERT** | **0.158** | **0.310** | **0.921** | **0.829** |
| Ensemble (LR & LGBM) | 0.450 | 0.455 | 0.732 | 0.727 |
| LightGBM + TF-IDF | 0.456 | 0.461 | 0.723 | 0.719 |
| Logistic Regression + TF-IDF | 0.469 | 0.477 | 0.720 | 0.711 |
| XGBoost + TF-IDF | 0.479 | 0.484 | 0.708 | 0.704 |
| LightGBM | 0.530 | 0.534 | 0.666 | 0.662 |
| Random Forest | 0.464 | 0.535 | 0.715 | 0.660 |
| XGBoost | 0.530 | 0.535 | 0.666 | 0.662 |
| Random Forest + TF-IDF | 0.551 | 0.555 | 0.675 | 0.668 |
| Logistic Regression | 0.584 | 0.585 | 0.614 | 0.614 |
| Baseline (DummyClassifier) | 0.658 | 0.658 | 0.488 | 0.488 |

**Best Model**: DistilBERT achieves state-of-the-art performance with the lowest validation log loss (0.310) and highest F1-score (0.829).

**Key Insights**:
- Transformer-based models significantly outperform traditional ML approaches
- TF-IDF features improve performance for most traditional models
- Ensemble methods provide better results than individual baseline models
- Adding TF-IDF to Random Forest actually decreased performance

## Technical Stack

- **Programming**: Python 3.x
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **NLP**: NLTK, Hugging Face transformers, scikit-learn (TF-IDF)
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch (via transformers)
- **Utilities**: joblib for model serialization

## Getting Started

### Running the Notebooks
Execute notebooks in sequence:

1. **EDA**: `01_data_EDA.ipynb` - Understand the dataset
2. **Preprocessing**: `02_preprocessing.ipynb` - Generate features
3. **Traditional ML**: `03_models.ipynb` - Train baseline models
4. **Deep Learning**: `04_distil_bert.ipynb` - Train transformer model

## Key Features

- **Comprehensive comparison** of traditional ML vs. deep learning approaches
- **Feature engineering** techniques for text similarity
- **TF-IDF vectorization** for semantic representation
- **Transformer fine-tuning** with DistilBERT
- **Ensemble methods** combining multiple models
- **Complete evaluation** with multiple metrics and error analysis

## Results Storage

- Model configurations saved in `model/` directory
- Evaluation metrics serialized in `results/` directory
- Processed datasets stored in `data/processed_data/`

## Trained Model
Due to GitHub file size limits, the fine-tuned BERT model is loaded to [hugging face](https://huggingface.co/zvonovska/quora-pairs-distilbert)

## Deployed API (FastAPI + Docker)

**Demo:** [quora_deployment](https://huggingface.co/spaces/zvonovska/quora-duplicated-questions-d)

The demo provides a simple web interface where users can enter two questions and receive a predicted probability of duplication in real time.


