# Fraud Detection Model - Project Information

## Overview

This project implements a machine learning model to detect fraudulent credit card transactions using logistic regression. The model is trained on a highly imbalanced dataset containing credit card transactions, where only a small fraction (0.172%) are fraudulent.

## Dataset

The dataset used is from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), containing:

- **284,807 transactions total**
- **492 fraudulent transactions** (0.172% of total)
- **30 features** (V1–V28 PCA-transformed, plus Time and Amount)
- **Binary target variable** (`Class`: 0 = legitimate, 1 = fraud)

## Key Challenges

- **Class Imbalance**: The dataset is highly imbalanced with very few fraud cases.  
- **Feature Engineering**: All features except Time and Amount are already PCA transformed.  
- **Model Performance**: Balancing accuracy for both classes despite imbalance.

## Methodology

1. **Data Exploration**  
   - Analyzed distribution of classes and feature statistics

2. **Handling Imbalance**  
   - Used under-sampling to balance the classes (492 legitimate vs. 492 fraud samples)

3. **Model Training**  
   - Implemented Logistic Regression with default parameters

4. **Evaluation**  
   - Assessed performance on both training and test sets

## Results

- **Training Accuracy**: 94.03%  
- **Testing Accuracy**: 92.89%

## Dependencies

- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

## How to Use

1. Clone the repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
