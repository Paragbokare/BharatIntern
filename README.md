# Bharat Intern - Data Science / Machine Learning Projects

## 🌟 Overview

This repository documents the work completed during the **Bharat Intern** Virtual Internship Program in the domain of Data Science and Machine Learning. It serves as a portfolio demonstrating skills in data analysis, feature engineering, model training, and evaluation for classification problems.

The repository includes two distinct projects:

1.  **Titanic Survival Prediction:** A classic classification problem to predict passenger survival using historical data.
2.  **SMS Classification (Spam/Ham):** A Natural Language Processing (NLP) task to classify text messages as spam or legitimate ('ham').

## 📁 Projects

### 1. Project: Titanic Survival Prediction

| File | `Titanic Survival.py` |
| :--- | :--- |

#### **Goal**
The objective of this project is to build a machine learning model that predicts whether a passenger survived the sinking of the Titanic based on various features such as age, gender, passenger class, and ticket fare.

#### **Technologies & Libraries**
* **Python**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib/Seaborn:** For data visualization and exploratory data analysis (EDA).
* **Scikit-learn:** For data preprocessing, model selection, and training.

#### **Methodology (High-Level)**
1.  **Exploratory Data Analysis (EDA):** Initial inspection of the data, visualization of feature distributions, and identification of correlations.
2.  **Data Preprocessing:** Handling missing values (e.g., Age, Cabin, Embarked), encoding categorical features (e.g., Sex, Embarked), and scaling numerical features.
3.  **Feature Engineering:** Creating new features (e.g., 'Family Size', 'Title' from Name) to improve model performance.
4.  **Model Training:** Training a classification model (e.g., Logistic Regression, Decision Tree, or Random Forest) on the processed training data.
5.  **Evaluation:** Assessing the model's performance using metrics like accuracy, precision, and recall.

---

### 2. Project: SMS Classification (Spam/Ham)

| File | `SMS classification.py` |
| :--- | :--- |

#### **Goal**
The objective of this NLP project is to classify incoming text messages (SMS) as either **Spam** (unwanted, unsolicited messages) or **Ham** (legitimate messages).

#### **Technologies & Libraries**
* **Python**
* **Pandas:** For loading and cleaning the text dataset.
* **NLTK (Natural Language Toolkit):** For text preprocessing (tokenization, stop-word removal, stemming/lemmatization).
* **Scikit-learn:** For converting text into numerical features (e.g., using **TF-IDF Vectorizer** or **Count Vectorizer**) and model training.
* **Classification Model:** Typically a Naive Bayes or Support Vector Machine (SVM) is used for this type of text classification.

#### **Methodology (High-Level)**
1.  **Data Loading and Cleaning:** Loading the dataset and performing initial text cleaning (removing punctuation, lowercasing).
2.  **Text Preprocessing:** Tokenizing the messages and removing common stop words (e.g., "the," "is," "a").
3.  **Feature Extraction:** Converting the clean text data into a numerical vector representation using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
4.  **Model Training:** Training a suitable classification model (e.g., Multinomial Naive Bayes) on the TF-IDF features.
5.  **Evaluation:** Evaluating the model's ability to correctly classify messages using accuracy, and metrics specific to imbalanced data like a confusion matrix.

---

## 🚀 Getting Started

To run these projects locally, follow the steps below.

### Prerequisites

Ensure you have Python installed. You will need the following libraries:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
