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
│   ├── app.py                # Flask backend API
│   ├── requirements.txt      # Backend dependencies
│   └── intent_oos_model/     # Model and config files
│
├── frontend/
│   ├── src/                  # React app source code
│   ├── package.json          # Frontend dependencies
│   └── public/               # Static assets
│
├── notebook/
│   └── Intent+OOS_detection.ipynb  # Model training and evaluation
│
├── model/                    # Optional saved models (if exported separately)
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

## Key Findings

- Sentence-BERT embeddings with Logistic Regression provide strong performance on intent classification.
- Confidence-based thresholding is effective for OOS detection without additional model components.
- Validation F1-score determines the optimal threshold, which generalizes well to the test set.

## Challenges

- Managing class imbalance between in-domain and OOS samples.
- Ensuring threshold stability across validation and test splits.
- Handling large model files in GitHub (use Git LFS or exclude in .gitignore).

## Notes

- The notebook may not render on GitHub due to metadata issues. To fix this, run:
  `jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --inplace notebook/Intent+OOS_detection.ipynb`
- All dependencies are listed in `backend/requirements.txt` and `frontend/package.json`.
