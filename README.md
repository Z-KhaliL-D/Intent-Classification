# Intent Classification and Out-of-Scope (OOS) Detection

## 1. Project Overview

This project implements an **Intent Classification and Out-of-Scope (OOS) Detection** system based on the CLINC OOS-eval dataset.
The main objective is to train a model that recognizes user intents accurately while also identifying queries that fall outside known categories (OOS).
The project demonstrates a complete machine learning workflow — from data preparation to model training, evaluation, and deployment through a Flask backend and React frontend.

Key components include:
- A **Jupyter Notebook** for training and evaluation.
- A **Flask API** serving predictions from the trained model.
- A **React web interface** for real-time testing of user inputs.

---

## 2. How to Run the Project

### 2.1 Folder Structure

```
project_root/
│
├── backend/
│   ├── app.py                 # Flask backend
│   ├── requirements.txt       # Backend dependencies
│   └── intent_oos_model/      # Trained model and configuration files
│
├── frontend/
│   ├── src/                   # React components
│   ├── public/                # Static assets
│   └── package.json           # Frontend dependencies
│
├── notebook/
│   └── Intent+OOS_detection.ipynb  # Model training and OOS analysis
│
├── .gitignore
└── README.md
```

### 2.2 Backend Setup

1. Navigate to the backend folder:
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate   # For Windows
   pip install -r requirements.txt
   ```

2. **IMPORTANT:**  
   Before running the backend, **unzip the trained model files** into a folder named:
   ```
   backend/intent_oos_model/
   ```

   Make sure this folder contains:
   ```
   sbert_model/
   intent_oos_model.pkl
   label_encoder.pkl
   oos_config.json
   ```

3. Start the Flask backend:
   ```bash
   python app.py
   ```

4. The backend will start locally at `http://127.0.0.1:5000`.

### 2.3 Frontend Setup

1. In a new terminal:
   ```bash
   cd frontend
   npm install
   npm start
   ```
2. The React app runs at `http://localhost:3000` and connects to the Flask API.

### 2.4 Model Training (Optional)

To retrain or fine-tune the model, open the notebook:
```
notebook/Intent+OOS_detection.ipynb
```

After training, export the model as:
```
sbert_model/
intent_oos_model.pkl
label_encoder.pkl
oos_config.json
```

Then zip these files if sharing, or unzip them into:
```
backend/intent_oos_model/
```

---

## 3. Detailed Approach

### 3.1 Data
- Dataset: **CLINC OOS-eval small version**.
- Contains multiple domains and intents, plus labeled OOS queries.
- Used combined in-domain and OOS samples for training, validation, and testing.

| Split | Examples |
|-------|-----------|
| Train | 7,500 |
| Validation | 3,000 |
| Test | 4,500 |

### 3.2 Preprocessing
- Combined OOS and in-domain samples per split.
- Extracted text and intent labels into DataFrames.
- Applied Label Encoding to map intent names to numeric IDs (keeping “oos” as a special label).
- Prepared clean text datasets for embedding.

### 3.3 Embedding
- Used **Sentence-BERT (`all-MiniLM-L6-v2`)** from Sentence-Transformers.
- Converts each sentence into a 384-dimensional vector embedding.
- Handles tokenization, casing, and punctuation internally.

### 3.4 Model
- Classifier: **Logistic Regression** (`max_iter=2000`)
- Input: SBERT sentence embeddings.
- Output: Intent label predictions.
- Achieved:
  - Validation Accuracy: **0.924**
  - Test Accuracy: **0.852**

### 3.5 OOS Detection Mechanism
The baseline classifier performs well on known intents but struggles to detect unseen (OOS) queries.
To fix this, we introduced a **confidence-based threshold** mechanism:

- The model’s maximum probability is treated as a confidence score.
- If the maximum probability is **below the threshold**, the query is marked as OOS.
- The optimal threshold is chosen based on F1-score on the validation set.

**Results:**

| Metric | Before Threshold | After Threshold |
|---------|------------------|-----------------|
| Precision | 0.84 | 0.70 |
| Recall | 0.59 | 0.91 |
| F1-score | 0.69 | 0.79 |

This improves recall significantly, making the system safer and better at flagging unfamiliar inputs.

---

## 4. Key Findings and Challenges

### Key Findings
- SBERT embeddings with Logistic Regression are highly effective for intent classification.
- Confidence thresholding greatly enhances OOS detection robustness.
- The approach achieves a strong balance between performance and simplicity.

### Challenges
- Handling poor OOS detection.
- Selecting an optimal threshold that generalizes well.

---

## 5. Conclusion and Future Work

This project presents a practical and interpretable approach for **Intent Classification with OOS Detection**.
By integrating Sentence-BERT embeddings with a simple Logistic Regression model and confidence-based thresholding, the system achieves reliable and safe performance for real-world applications.

Future extensions could include:
- Fine-tuning transformer models directly on OOS data.
- Experimenting with ensemble classifiers or neural networks.
- Evaluating on larger, more diverse conversational datasets.
