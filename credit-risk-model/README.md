## Credit Scoring Business Understanding

### 1. Basel II and Model Explainability

Basel II is an international banking regulation that requires banks to measure and manage credit risk responsibly. One of its key principles is that banks must be able to explain and justify the models they use for credit decisions.

- **Why is this important?**
  - Regulators want to ensure banks aren't taking on hidden risks that could threaten financial stability. Basel II requires banks to hold enough capital to cover their risks, and this is only possible if risks are measured transparently.
  - If a model is a "black box," it's hard to know if it's fair, biased, or making mistakes. For example, a highly complex model might inadvertently discriminate against certain groups, or make decisions based on spurious correlations.
  - Banks must document how the model works, what data it uses, and why it makes certain predictions. This includes keeping records of model development, validation, and ongoing monitoring.
  - This transparency protects both the bank and its customers, and allows for audits and regulatory reviews. Regulators may require banks to justify individual credit decisions, especially if a customer is denied credit.
  - Basel II also emphasizes the need for stress testing and scenario analysis, which are easier to perform and interpret with simpler, more transparent models.

**In summary:** Basel II pushes banks to use models that are not just accurate, but also interpretable, transparent, and well-documented. This means you often need to favor models and features that can be explained to both regulators and business stakeholders. For example, a logistic regression model with clear, documented features is easier to defend than a complex neural network.

---

### 2. Importance and Risk of Proxy Variables

In real-world data, you often don't have a perfect label for "default" (e.g., maybe you don't know for sure if a customer would have defaulted, or you only have partial payment data). So, you create a proxy variable — a stand-in label based on observable behavior (like missed payments, inactivity, or cancellations).

- **Why do we need a proxy?**
  - To train a model, you need a target variable (good/bad). In the absence of a direct "default" label, proxies allow you to approximate the outcome of interest.
  - If you don't have a direct "default" label, you use the best available signal. For example, you might define a "bad" customer as someone who missed two consecutive payments, or who was inactive for a certain period after receiving credit.
  - Proxies enable you to leverage available data to build and validate your models, even if the data is imperfect.

- **What could go wrong?**
  - The proxy might not perfectly match true default risk. For example, a customer who missed a payment due to a temporary issue but later paid in full might be incorrectly labeled as "bad."
  - You could mislabel customers, leading to unfair or inaccurate credit decisions. This can result in denying credit to good customers or extending credit to risky ones.
  - This can lead to biased models, unfairly denying credit, or exposing the bank to more risk. If the proxy is systematically biased (e.g., it penalizes certain customer groups), the model will inherit this bias.
  - Over-reliance on proxies can also make it harder to compare model performance across different datasets or time periods, as the definition of "default" may change.

**In summary:** Proxy variables are necessary when direct labels are missing, but they introduce the risk of label noise and misclassification, which can impact both model performance and fairness. It's important to carefully define and validate proxies, and to be transparent about their limitations. Regularly reviewing and updating proxy definitions as more data becomes available is also a best practice.

---

### 3. Trade-Off: Simple vs Complex Models

- **Simple, Explainable Models** (e.g., Logistic Regression, Weight of Evidence):
  - **Pros:**
    - Easy to interpret, explain, and document. Each feature's impact on the prediction is clear and can be visualized or explained to non-technical stakeholders.
    - Regulators and business users can understand how decisions are made, which builds trust and facilitates compliance.
    - Simpler models are less prone to overfitting and are easier to monitor and maintain over time.
    - Example: A logistic regression model might show that recent missed payments increase risk, while high transaction frequency reduces it.
  - **Cons:**
    - May not capture complex patterns in the data, potentially less accurate if the underlying relationships are non-linear or involve interactions between features.
    - Limited flexibility in handling large numbers of features or unstructured data.

- **Complex, Powerful Models** (e.g., Gradient Boosting, Neural Networks):
  - **Pros:**
    - Can achieve higher predictive accuracy by modeling complex relationships and interactions in the data.
    - Useful when there are many features or when the data is high-dimensional or unstructured (e.g., text, images).
    - Example: A gradient boosting model might uncover subtle patterns in customer behavior that a simple model would miss.
  - **Cons:**
    - Harder to interpret ("black box"), more difficult to document, and may not be accepted by regulators who require clear explanations for credit decisions.
    - Increased risk of overfitting, especially if not properly validated.
    - More complex to implement, monitor, and update.
    - May require additional tools (e.g., SHAP, LIME) to provide post-hoc explanations, which can add complexity and may not fully satisfy regulatory requirements.

**In summary:** There's a trade-off between accuracy and interpretability. In regulated industries like banking, explainability often takes priority — even if it means sacrificing some predictive power. The choice of model should balance predictive performance with the need for transparency, fairness, and regulatory compliance. In many cases, a slightly less accurate but more interpretable model is preferred for credit scoring applications. 