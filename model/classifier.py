import joblib
from sklearn.dummy  import DummyClassifier
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix,average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
import torch
import pandas as pd
import numpy as np
from .data import load_data , data_encode, split_data, impute_and_standardize, preprocess_data
import argparse


torch.manual_seed(42)

x, y = load_data()
y = data_encode(y)
x_train, y_train, x_val, y_val = split_data(x, y)
x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)
x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)

def trivial_baseline_classifier():
    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(x_train, y_train)
    dummy_predictions = dummy_clf.predict(x_val)
    accuracy = accuracy_score(y_val, dummy_predictions)
    recall = recall_score(y_val, dummy_predictions)
    precision = precision_score(y_val, dummy_predictions)
    f1 = f1_score(y_val, dummy_predictions)
    return dummy_clf, accuracy, recall, precision, f1

dummy_clf, accuracy, recall, precision, f1 = trivial_baseline_classifier()


print(f"Trivial Baseline Classifier Accuracy: {accuracy:.4f}")
print(f"Trivial Baseline Classifier Recall: {recall:.4f}")
print(f"Trivial Baseline Classifier Precision: {precision:.4f}")
print(f"Trivial Baseline Classifier F1: {f1:.4f}")
print(confusion_matrix(y_val, dummy_clf.predict(x_val)))


def gradient_boost_classifier(random_state=None):
    classifier = GradientBoostingClassifier(random_state=random_state)
    classifier.fit(x_train, y_train)
    gradient_probabilities = classifier.predict_proba(x_val)[:,1]
    thresholds = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1]
    threshold_results = []

    for threshold in thresholds:
        predictions = (gradient_probabilities >= threshold).astype(int)
        false_positive = np.sum((predictions == 1) & (y_val.numpy() == 0))
        false_negative = np.sum((predictions == 0) & (y_val.numpy() == 1))
        total_cost = false_positive * 10 + false_negative * 500
        gradient_accuracy = accuracy_score(y_val, predictions)
        gradient_recall = recall_score(y_val, predictions)
        gradient_precision = precision_score(y_val, predictions)
        gradient_f1 = f1_score(y_val, predictions)

        threshold_results.append({
            'threshold': threshold,
            'accuracy': gradient_accuracy,
            'recall': gradient_recall,
            'precision': gradient_precision,
            'f1': gradient_f1,
            'total_cost': total_cost
            }
        )
    best_result = min(threshold_results, key=lambda result: result['total_cost'], default=None)

    return classifier, threshold_results, gradient_probabilities,best_result,impute, standardize

classifier, threshold_results, gradient_probabilities,best_result,impute, standardize = gradient_boost_classifier()

print("gradient boosting classifier results for different thresholds:")
for result in threshold_results:
    print(f"Threshold: {result['threshold']}, Accuracy: {result['accuracy']:.4f}, Recall: {result['recall']:.4f}, Precision: {result['precision']:.4f}, F1: {result['f1']:.4f}, Total Cost: {result['total_cost']:.2f}")

print(f"best threshold with the lowest total cost: {best_result['threshold']}, Total Cost: {best_result['total_cost']:.2f}")

print("confusion matrix for gradient boosting classifier at different thresholds:")
for result in threshold_results:
    threshold = result['threshold']
    predictions = (gradient_probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_val, predictions)
    print(f"Threshold: {threshold}")
    print(matrix)
    

def feature_importance(classifier):
    feature_importances = classifier.feature_importances_
    feature_importances = pd.DataFrame(feature_importances, columns=['importance'], index=x.columns).sort_values(by='importance', ascending=False).head(10)
    return feature_importances

feature_importances = feature_importance(classifier)

def top_features_(feature_importances):
    top_features = feature_importances.index.tolist()
    top_features_count = x[top_features].isnull().sum()
    top_features_percentage = x[top_features].isnull().mean() * 100
    return top_features , top_features_count, top_features_percentage


top_features, top_features_count, top_features_percentage = top_features_(feature_importances)

print(classifier.feature_importances_.shape)
print(classifier.feature_importances_)
print(feature_importances)
print(top_features)
print(top_features_count)
print(top_features_percentage)


def average_precision_at_three_seeds():
    seeds = [42, 67,69]
    average_precisions = []

    for seed in seeds:
        random_state=seed
        classifier, _, gradient_probabilities,best_result,impute, standardize = gradient_boost_classifier(random_state)
        average_precision = average_precision_score(y_val, gradient_probabilities)
        average_precisions.append(average_precision)

    mean_ap =np.average(average_precisions)

    print(f"Average precision scores for seeds {seeds}: {average_precisions}")
    print(f"average precision at three seeds: {mean_ap:.4f}")

    return mean_ap

mean_ap = average_precision_at_three_seeds()

gradient_boost_classifier = joblib.dump(classifier, 'gradient_boost_classifier.pkl')
gradient_boost_threshold = joblib.dump(best_result['threshold'], 'gradient_boost_threshold.pkl')
impute_and_standardize = joblib.dump((impute, standardize), 'impute_and_standardize.pkl')
