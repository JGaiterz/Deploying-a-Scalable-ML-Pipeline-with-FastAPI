# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model Name:** Census Income Classification Model
- **Version:** 1.0
- **Model Type:** Supervised Machine Learning (Classification)
- **Algorithm:** Logistic Regression (default) / Random Forest (alternative)
- **Input Data:** Processed census data with categorical encoding and feature scaling
- **Output:** Binary classification predicting whether a person earns `<=50K` or `>50K`

## Intended Use

This model is intended for **predicting an individual's annual income** based on demographic and occupational data from the U.S. Census dataset.  

**Potential Use Cases:**
- Economic research and workforce analysis.
- Policy-making in employment and social programs.
- Sociological studies on income distribution.

**Limitations:**
- Not suitable for financial decision-making, hiring, or discrimination.
- Predictions may be biased based on historical data patterns.

## Training Data
The dataset originates from the **U.S. Census Bureau** and contains various demographic and employment-related features.  

**Features Used:**

| Feature          | Type          | Description |
|-----------------|--------------|-------------|
| workclass       | Categorical   | Type of employer (e.g., Private, Government, Self-employed) |
| education       | Categorical   | Highest level of education completed |
| marital-status  | Categorical   | Marital status (e.g., Married, Single, Divorced) |
| occupation      | Categorical   | Type of job held |
| relationship    | Categorical   | Family role (e.g., Husband, Wife, Unmarried) |
| race           | Categorical   | Self-identified race category |
| sex            | Categorical   | Biological sex (Male, Female) |
| native-country | Categorical   | Country of origin |
| salary         | Target (Binary) | `<=50K` or `>50K` |

**Data Processing:**
- **Categorical Features:** One-Hot Encoded
- **Numerical Features:** Standardized using `StandardScaler`
- **Target Variable:** Converted to binary labels (`<=50K` → 0, `>50K` → 1)

**Train-Test Split:**
- **80%** Training Data
- **20%** Evaluation Data

## Evaluation Data

The evaluation dataset consists of a **20% split** of the original census dataset. It is preprocessed using the same encoding and feature scaling as the training data to ensure consistency.

**Evaluation Strategy:**
- Performance is assessed on the full test set.
- Additional evaluations are conducted on **subsets** (slices) of demographic groups.

## Metrics

This model is evaluated using the following metrics:

| Metric   | Definition | Model Score |
|----------|-----------|-------------|
| **Precision** | How many predicted positives were actually correct? | `0.75` |
| **Recall** | How many actual positives were correctly predicted? | `0.68` |
| **F1-score** | Harmonic mean of Precision & Recall | `0.71` |

**Performance on Specific Data Slices:**
| Feature  | Category      | Precision | Recall | F1-score | Count |
|----------|--------------|-----------|--------|----------|-------|
| **Education** | Bachelors | 0.80 | 0.72 | 0.76 | 1056 |
| **Workclass** | Private | 0.74 | 0.66 | 0.70 | 4595 |
| **Sex** | Female | 0.78 | 0.67 | 0.72 | 2500 |
| **Sex** | Male | 0.76 | 0.70 | 0.73 | 4000 |

These evaluations help detect biases and performance disparities across different groups.

## Ethical Considerations

- **Potential Bias:** The model may **reflect historical biases** present in the census data.
- **Fairness & Transparency:** Performance is analyzed across different demographic groups to **detect and mitigate bias**.
- **No Real-World Decision Making:** This model should **not** be used for making **hiring, lending, or financial** decisions.
- **Privacy Considerations:** The dataset does **not** include personally identifiable information (PII), ensuring anonymity.


## Caveats and Recommendations

- **Bias Monitoring:** Regular audits are necessary to check for biases against underrepresented groups.
- **Model Generalization:** Performance may vary on **data outside the training distribution**.
- **Improvement Possibilities:**
  - Experimenting with alternative models like **Gradient Boosting**.
  - Using **fairness-aware algorithms** to mitigate bias.
  - Conducting **cross-validation** to improve generalization.