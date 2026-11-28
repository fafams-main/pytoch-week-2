"""Command-line entry point for the APS failure project.

Run from the project root (Pytorch-week-2), e.g.:

    py -m model.cli --smoke
    py -m model.cli --train --seed 67
    py -m model.cli --evaluate --checkpoint best_model_42.pth --metrics-out metrics.json
"""
import argparse
import json

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from .data import load_test_data, data_encode_test, preprocess_test_data
from .train import APSClassifier

FP_COST = 10
FN_COST = 500


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="APS Failure Prediction")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run a short training test (2 epochs)")
    mode.add_argument("--train", action="store_true", help="Run full training")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate a saved checkpoint on the test set")

    parser.add_argument("--checkpoint", type=str, default="best_model_42.pth", help="Path to the model checkpoint")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: load train_test_threshold.pkl)")
    parser.add_argument("--metrics-out", type=str, default=None, help="Path to save evaluation metrics as JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=120, help="Maximum training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_model_{seed}.pth")

    return parser.parse_args(argv)


def run_training(args, epochs):
    # imported here so the dataloaders are only built when training
    from .dataloader import train_loader, val_loader

    torch.manual_seed(args.seed)
    model = APSClassifier()
    model.fit(train_loader, val_loader, epochs=epochs,
              resume_train=args.resume, random_state=args.seed)
    return model


def run_evaluation(args):
    torch.manual_seed(args.seed)

    model = APSClassifier()
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    threshold = args.threshold if args.threshold is not None else joblib.load("train_test_threshold.pkl")

    # reuse the imputer/standardizer fitted on the training data
    impute, standardize = joblib.load("impute_and_standardize.pkl")
    x_test, y_test = load_test_data()
    y_test = data_encode_test(y_test)
    x_test, y_test = preprocess_test_data(x_test, y_test, impute, standardize)

    x_tensor = torch.as_tensor(np.asarray(x_test), dtype=torch.float32)

    with torch.no_grad():
        probabilities = F.softmax(model(x_tensor), dim=1)[:, 1].cpu().numpy()

    y_true = np.asarray(y_test)
    predictions = (probabilities >= threshold).astype(int)
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))

    return {
        "checkpoint": args.checkpoint,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions),
        "average_precision": average_precision_score(y_true, probabilities),
        "false_positives": fp,
        "false_negatives": fn,
        "total_cost": fp * FP_COST + fn * FN_COST,
    }


def main(argv=None):
    args = parse_args(argv)
    print(args)

    if args.smoke:
        run_training(args, epochs=2)
        print("Smoke test finished.")
    elif args.train:
        run_training(args, epochs=args.epochs)
    elif args.evaluate:
        metrics = run_evaluation(args)
        for name, value in metrics.items():
            print(f"{name}: {value}")
        if args.metrics_out:
            with open(args.metrics_out, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
