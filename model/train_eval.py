import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from dataloader import train_loader, val_loader
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix,average_precision_score
from data import load_data,preprocess_data,split_data,impute_and_standardize,data_encode


from train import APSClassifier

x, y = load_data()
y = data_encode(y)
x_train, y_train, x_val, y_val = split_data(x, y)
x_train, x_val, impute, standardize = impute_and_standardize(x_train, x_val)
x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)


def predict_train_proba(model, val_loader):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            predictions = model(x_batch)
            probabilities.append(F.softmax(predictions, dim=1)[:, 1].cpu().numpy())
    print("predict shape",predictions.shape)
    print("probability shape",F.softmax(predictions, dim=1)[:, 1].shape)
    return np.concatenate(probabilities)




def train_threshold_classifier(random_state):
    torch.manual_seed(random_state)
    model = APSClassifier()
    train_losses, val_losses, best_val_loss, best_epoch, epochs,epoch,average_train_loss, average_val_loss = model.fit(train_loader, val_loader, epochs=120,resume_train=True,random_state=random_state)

    model_probabilities =(predict_train_proba(model, val_loader))
    thresholds = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1]
    train_threshold_results = []
    
    for threshold in thresholds:
        predictions = (model_probabilities >= threshold).astype(int)
        false_positive = np.sum((predictions == 1) & (y_val.numpy() == 0))
        false_negative = np.sum((predictions == 0) & (y_val.numpy() == 1))
        total_cost = false_positive * 10 + false_negative * 500
        train_accuracy = accuracy_score(y_val, predictions)
        train_recall = recall_score(y_val, predictions)
        train_precision = precision_score(y_val, predictions)
        train_f1 = f1_score(y_val, predictions)

        train_threshold_results.append({
            'threshold': threshold,
            'accuracy': train_accuracy,
            'recall': train_recall,
            'precision': train_precision,
            'f1': train_f1,
            'total_cost': total_cost
            }
        )
        
    best_result = min(train_threshold_results, key=lambda result: result['total_cost'], default=None)

    return model, train_threshold_results, model_probabilities, y_val, best_result

def train_average_precision_at_three_seeds():
    seeds = [42, 67,69]
    average_precisions = []

    for seed in seeds:
        random_state=seed
        model, _, model_probabilities, y_val, best_result = train_threshold_classifier(random_state)
        average_precision = average_precision_score(y_val, model_probabilities)
        average_precisions.append(average_precision)

    mean_ap =np.average(average_precisions)

    print(f"Average precision scores for seeds {seeds}: {average_precisions}")
    print(f"average precision at three seeds: {mean_ap:.4f}")

    return mean_ap

mean_ap = train_average_precision_at_three_seeds()

def main():
    model, threshold_results, model_probabilities,y_val, best_result = train_threshold_classifier(random_state=42)

    print("APS Classifier results for different thresholds:")
    for result in threshold_results:
        print(f"Threshold: {result['threshold']}, Accuracy: {result['accuracy']:.4f}, Recall: {result['recall']:.4f}, Precision: {result['precision']:.4f}, F1: {result['f1']:.4f}, Total Cost: {result['total_cost']:.2f}")

    print("confusion matrix for model at different thresholds:")
    for result in threshold_results:
        threshold = result['threshold']
        predictions = (model_probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_val, predictions)
        print(f"Threshold: {threshold}")
        print(matrix)

    print(f"best threshold with the lowest total cost: {best_result['threshold']}, Total Cost: {best_result['total_cost']:.2f}")

def plot_losses(train_losses, val_losses, best_epoch, save_path=None):
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, train_losses, label="Train loss")
    plt.plot(epochs_range, val_losses, label="Val loss")

    plt.axvline(best_epoch + 1, color="gray", linestyle="--", label=f"Best epoch ({best_epoch + 1})")
    plt.scatter(best_epoch + 1, val_losses[best_epoch], color="red", zorder=5)

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    plot_losses(train_losses, val_losses, best_epoch, save_path="loss_curve.png")


def plot_metric(values, best_epoch, name):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(values) + 1), values, label=name)
    plt.axvline(best_epoch + 1, color="gray", linestyle="--", label="Best-loss epoch")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()