# Insulin Dosage Model


## Objective

Train a model that can learn from past glucose, carb, and insulin administration data to:
- Predict the appropriate **short-acting and long-acting insulin dosages** based on current physiological context
- Improve dosage predictions using **reinforcement learning** based on predicted vs. actual glucose outcomes

## Core Components

### 1. `glucose_insulin_model_testing`

A simplified LSTM-based pipeline for:
- Preprocessing CGM data
- Feature engineering (e.g., sinusoidal time-of-day encodings)
- Building input/output windows for sequence modeling
- Training an LSTM model to predict insulin dosages

---

### 2. `insulin_dosage_model`

An advanced pipeline with:
- GRU-based model architecture
- **Optuna** hyperparameter tuning for model optimization
- **MinMax scaling**, moving averages, and time-of-day embedding
- **Reinforcement learning** component using **Vowpal Wabbit** to iteratively improve dose policies

---

## Data

Expected CSV columns:
- `Timestamp (YYYY-MM-DDThh:mm:ss)`
- `Glucose Value (mg/dL)`
- `Insulin Value (u)`
- `Event Subtype` (e.g., "Long-Acting", "Fast-Acting")
- `Carb Value (grams)`

---

## Dependencies

Install via pip:

```bash
pip install tensorflow pandas numpy matplotlib optuna vowpalwabbit
