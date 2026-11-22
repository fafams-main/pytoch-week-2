from sklearn.dummy  import DummyClassifier
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import torch
import pandas as pd
import numpy as np
from dataloader import train_loader, val_loader
from data import load_data , data_encode, split_data, impute_and_standardize, preprocess_data


torch.manual_seed(42)

x, y = load_data()
y = data_encode(y)
x_train, y_train, x_val, y_val = split_data(x, y)
x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)
x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)

def dummy_classifier():
    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(x_train, y_train)
    dummy_predictions = dummy_clf.predict(x_val)
    accuracy = accuracy_score(y_val, dummy_predictions)
    recall = recall_score(y_val, dummy_predictions)
    precision = precision_score(y_val, dummy_predictions)
    f1 = f1_score(y_val, dummy_predictions)
    return dummy_clf, accuracy, recall, precision, f1

dummy_clf, accuracy, recall, precision, f1 = dummy_classifier()


print(f"Dummy Classifier Accuracy: {accuracy:.4f}")
print(f"Dummy Classifier Recall: {recall:.4f}")
print(f"Dummy Classifier Precision: {precision:.4f}")
print(f"Dummy Classifier F1: {f1:.4f}")
print(confusion_matrix(y_val, dummy_clf.predict(x_val)))

def gradient_boost_classifier():
    classifier = GradientBoostingClassifier()
    classifier.fit(x_train, y_train)
    gradient_predictions = classifier.predict(x_val)
    gradient_accuracy = accuracy_score(y_val, gradient_predictions)
    gradient_recall = recall_score(y_val, gradient_predictions)
    gradient_precision = precision_score(y_val, gradient_predictions)
    gradient_f1 = f1_score(y_val, gradient_predictions)
    return classifier, gradient_accuracy, gradient_recall, gradient_precision, gradient_f1, gradient_predictions

classifier, gradient_accuracy, gradient_recall, gradient_precision, gradient_f1,gradient_predictions = gradient_boost_classifier()

print(f"Gradient Boosting Classifier Accuracy: {gradient_accuracy:.4f}")
print(f"Gradient Boosting Classifier Recall: {gradient_recall:.4f}")
print(f"Gradient Boosting Classifier Precision: {gradient_precision:.4f}")
print(f"Gradient Boosting Classifier F1: {gradient_f1:.4f}")
print(confusion_matrix(y_val, gradient_predictions))

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
