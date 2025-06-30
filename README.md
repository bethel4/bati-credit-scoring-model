# bati-credit-scoring-model
Credit Scoring ML system for Bati Bank's BNPL service

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Credit Scoring Business Understanding

### 1. Basel II and Model Explainability

Basel II is an international banking regulation that requires banks to measure and manage credit risk responsibly. One of its key principles is that banks must be able to explain and justify the models they use for credit decisions.

- **Why is this important?**
  - Regulators want to ensure banks aren't taking on hidden risks.
  - If a model is a "black box," it's hard to know if it's fair, biased, or making mistakes.
  - Banks must document how the model works, what data it uses, and why it makes certain predictions.
  - This transparency protects both the bank and its customers, and allows for audits and regulatory reviews.

**In summary:** Basel II pushes banks to use models that are not just accurate, but also interpretable, transparent, and well-documented. This means you often need to favor models and features that can be explained to both regulators and business stakeholders.

---

### 2. Importance and Risk of Proxy Variables

In real-world data, you often don't have a perfect label for "default" (e.g., maybe you don't know for sure if a customer would have defaulted, or you only have partial payment data). So, you create a proxy variable — a stand-in label based on observable behavior (like missed payments, inactivity, or cancellations).

- **Why do we need a proxy?**
  - To train a model, you need a target variable (good/bad).
  - If you don't have a direct "default" label, you use the best available signal.

- **What could go wrong?**
  - The proxy might not perfectly match true default risk.
  - You could mislabel customers (e.g., someone who missed a payment but later paid in full).
  - This can lead to biased models, unfairly denying credit, or exposing the bank to more risk.

**In summary:** Proxy variables are necessary when direct labels are missing, but they introduce the risk of label noise and misclassification, which can impact both model performance and fairness.

---

### 3. Trade-Off: Simple vs Complex Models

- **Simple, Explainable Models** (e.g., Logistic Regression, Weight of Evidence):
  - **Pros:** Easy to interpret, explain, and document. Regulators and business users can understand how decisions are made.
  - **Cons:** May not capture complex patterns in the data, potentially less accurate.

- **Complex, Powerful Models** (e.g., Gradient Boosting, Neural Networks):
  - **Pros:** Can achieve higher predictive accuracy by modeling complex relationships.
  - **Cons:** Harder to interpret ("black box"), more difficult to document, and may not be accepted by regulators.

**In summary:** There's a trade-off between accuracy and interpretability. In regulated industries like banking, explainability often takes priority — even if it means sacrificing some predictive power. 