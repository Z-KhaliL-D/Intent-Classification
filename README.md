# Intent Classification and Out-of-Scope (OOS) Detection

## Overview

This project implements an Intent Classification and Out-of-Scope (OOS) Detection system using Sentence-BERT (SBERT) embeddings and a Logistic Regression classifier. 
The model is trained to recognize user intents from text queries and detect queries that are out-of-scope (OOS).

The system includes:
- A Jupyter notebook for data preparation, training, and OOS threshold optimization
- A Flask backend API to serve predictions
- A simple React frontend for testing and demonstration

## Approach

1. **Data**
   - Used the CLINC OOS-eval dataset.
   - Combined in-domain and OOS samples for training, validation, and testing.

2. **Embeddings and Model**
   - Used `all-MiniLM-L6-v2` from Sentence-Transformers to create text embeddings.
   - Trained a Logistic Regression classifier with `max_iter=2000`.

3. **OOS Detection**
   - Used classifier probability confidence as a signal for OOS detection.
   - Determined an optimal threshold on the validation set that maximizes F1-score.
   - If the model’s max class probability is below the threshold, the query is flagged as OOS.

4. **Evaluation**
   - Reported accuracy and classification report for intent prediction.
   - Reported precision, recall, and F1 for binary OOS detection.

## Repository Structure

```
project_root/
│
├── backend/
│   ├── app.py                
│   ├── requirements.txt     
│   └── intent_oos_model/     
│
├── frontend/
│   ├── src/                  
│   ├── package.json          
│   └── public/               
│
├── notebook/
│   └── Intent+OOS_detection.ipynb  # Model training and evaluation
│               
├── .gitignore
└── README.md
```

## How to Run

### 1. Training the Model (Optional)
Run the Jupyter notebook located at `notebook/Intent+OOS_detection.ipynb` to retrain and export the model.

After training, the model files are saved as:
```
intent_oos_model.pkl
label_encoder.pkl
oos_config.json
```
You can zip and download these for deployment.

### 2. Flask Backend Setup

```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The backend will start on `http://127.0.0.1:5000`.

### 3. React Frontend Setup

```
cd frontend
npm install
npm start
```

The frontend runs on `http://localhost:3000` and communicates with the Flask API.


### 3. Approach

#### 3.1 Preprocessing
- Combined in-domain and OOS splits for train, validation, and test.
- Extracted text and intent labels into pandas DataFrames.
- Applied label encoding to map intent names into numeric IDs (keeping "oos" for rejection).
- Prepared clean datasets for Sentence-BERT embeddings.

#### 3.2 Tokenization
Sentence-BERT handles tokenization internally by splitting text into subword tokens and converting them into embeddings. It manages casing, punctuation, and subword units automatically. No additional preprocessing was needed, providing consistent semantic representations for all datasets.

#### 3.3 Model & Training
- Model: Logistic Regression with max_iter=2000
- Input: SBERT sentence embeddings for train, validation, and test sets
- Output: Predicted intent class for each query

Training achieved:
Validation Accuracy: 0.924
Test Accuracy: 0.852

#### 3.4 Evaluation
The baseline model achieves high accuracy on in-domain intents but struggles with OOS detection. This means the system might confidently misclassify unknown queries as valid intents, leading to poor user experience.

OOS detection results before improvement:
Precision: 0.84
Recall: 0.59
F1-score: 0.69

### 4. Improvement: Confidence Thresholding for OOS Detection
Instead of always accepting the top prediction, we use the model’s maximum probability as a confidence score. If this score is below a chosen threshold, the query is marked as OOS. The threshold is tuned on the validation set by sweeping multiple values and selecting the one with the best F1 score.

This method improves reliability by lowering false in-domain predictions and achieving a better precision–recall balance.

Comparison:

Before Thresholding
Precision: 0.84
Recall: 0.59
F1-score: 0.69

After Thresholding
Precision: 0.70
Recall: 0.91
F1-score: 0.79

### 5. Conclusion & Future Perspectives
A simple yet effective intent classification pipeline was developed using SBERT embeddings and Logistic Regression. Adding confidence-based thresholding for OOS detection improved robustness, reducing false in-domain predictions and achieving a better precision-recall tradeoff. 

Future improvements include fine-tuning transformer models and testing on larger or noisier datasets to better reflect real-world conditions.
