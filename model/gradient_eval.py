import joblib
from .classifier import gradient_boost_classifier
from .data import load_data, data_encode, split_data, impute_and_standardize, preprocess_data, load_test_data, data_encode_test, preprocess_test_data
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix,average_precision_score
import numpy as np

x, y = load_data()
y = data_encode(y)
x_train, y_train, x_val, y_val = split_data(x, y)
x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)
x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)
x_test, y_test = load_test_data()
y_test = data_encode_test(y_test)

impute, standardize = joblib.load('impute_and_standardize.pkl')

x_test, y_test = preprocess_test_data(x_test, y_test,impute, standardize)


classifier = joblib.load('gradient_boost_classifier.pkl')
threshold = joblib.load('gradient_boost_threshold.pkl')


def predict(classifier, x_test, threshold):
    probabilities = classifier.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions

def evaluate_gradient_model(classifier, x_test, y_test, threshold):
    predictions = predict(classifier, x_test, threshold)
    gradient_prediction_probabilities = classifier.predict_proba(x_test)[:, 1]
    false_positive = np.sum((predictions == 1) & (y_test.numpy() == 0))
    false_negative = np.sum((predictions == 0) & (y_test.numpy() == 1))
    total_cost = false_positive * 10 + false_negative * 500
    gradient_test_accuracy = accuracy_score(y_test, predictions)
    gradient_test_recall = recall_score(y_test, predictions)
    gradient_test_precision = precision_score(y_test, predictions)
    gradient_test_f1 = f1_score(y_test, predictions)

    return gradient_prediction_probabilities, gradient_test_accuracy, gradient_test_recall, gradient_test_precision, gradient_test_f1, total_cost

gradient_prediction_probabilities, gradient_test_accuracy, gradient_test_recall, gradient_test_precision, gradient_test_f1, total_cost = evaluate_gradient_model(classifier, x_test, y_test, threshold)

print(f"Gradient Boosting Classifier Test Accuracy: {gradient_test_accuracy:.4f}")
print(f"Gradient Boosting Classifier Test Recall: {gradient_test_recall:.4f}")
print(f"Gradient Boosting Classifier Test Precision: {gradient_test_precision:.4f}")
print(f"Gradient Boosting Classifier Test F1: {gradient_test_f1:.4f}")
print(f"Gradient Boosting Classifier Test Total Cost: {total_cost:.2f}")

