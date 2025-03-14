import pytest
# TODO: add necessary import
import pytest
import numpy as np
import pandas as pd
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    
    """
    Test if the train_model function returns the expected model type.
    """

    # Create dummy training data
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=(100,))
    
    # Train a logistic regression model
    model = train_model(X_train, y_train, model_type="logistic")
    assert isinstance(model, LogisticRegression), "train_model should return a LogisticRegression model."

    # Train a random forest model
    model_rf = train_model(X_train, y_train, model_type="random_forest")
    assert isinstance(model_rf, RandomForestClassifier), "train_model should return a RandomForestClassifier model."
       

# TODO: implement the second test. Change the function name and input as needed
def test_two():
    
    """
    Test if the compute_model_metrics function returns float values.
    """
    # Dummy ground truth and predictions
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Ensure metrics are floats
    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(fbeta, float), "F1-score should be a float."



# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test if process_data correctly processes input data and maintains expected dimensions.
    """
    # Create a dummy dataset
    data = pd.DataFrame({
        "workclass": ["Private", "Self-emp", "Government"],
        "education": ["Bachelors", "Masters", "PhD"],
        "salary": [">50K", "<=50K", ">50K"]
    })

    categorical_features = ["workclass", "education"]
    label = "salary"

    # Process the data
    X, y, encoder, lb = process_data(data, categorical_features, label, training=True)

    # Ensure X and y have expected shapes
    assert X.shape[0] == data.shape[0], "Processed features should have the same number of rows as input data."
    assert len(y) == data.shape[0], "Processed labels should match input data size."
  