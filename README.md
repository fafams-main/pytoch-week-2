# APS Failure Prediction Using PyTorch

## 1. Project Overview

This project focuses on predicting failures in Scania trucks using the **APS Failure and Operational Data dataset**.

The goal is to develop a machine learning model that can identify whether a truck is likely to experience an APS (Air Pressure System) failure based on its operational sensor data.

The main challenge is that failures are extremely rare compared with normal operating conditions. This creates a class imbalance problem, where a model can achieve high accuracy simply by predicting that most trucks will not fail.

Therefore, this project focuses on identifying failures while considering the different costs of false positives and false negatives.

**Main objective:** Build and evaluate a neural network using PyTorch, compare it against simpler machine learning approaches, and determine whether it provides a useful improvement for failure prediction.

---

# Part A: Results

## 2. Final Model Comparison

The following table compares the best reported validation results for the PyTorch neural network and Gradient Boosting classifier.

### Best results

| Metric         | PyTorch Neural Network | Gradient Boosting |
| -------------- | ---------------------: | ----------------: |
| Best threshold |                   0.03 |             0.005 |
| Accuracy       |                 97.45% |            95.79% |
| Recall         |                 93.20% |            96.00% |
| Precision      |                 38.96% |            27.84% |
| F1 Score       |                 0.5495 |            0.4317 |
| Total Cost     |                 12,150 |        **11,220** |

*The values above use each model's reported lowest-cost threshold.*

### Best model summary

| Model | Threshold | Accuracy | Recall | Precision | F1 | Average Precision across three seeds | Total Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.005 | 0.9607 | 0.9680 | 0.3708 | 0.5362 | 0.8114 | 12160.00 |
| PyTorch Neural Network | 0.03 | 0.9778 | 0.9240 | 0.4239 | 0.5811 | 0.8189 | 12640.00 |

### Average Precision Across Three Seeds

To check how consistently each model performs, both models were evaluated using seeds 42, 67, and 69.

| Model                  | Average Precision |
| ---------------------- | ----------------: |
| PyTorch Neural Network |        **0.8220** |
| Gradient Boosting      |            0.8114 |

### Comparison and Analysis

- **Accuracy:** The neural network performed better, achieving 97.45% compared with Gradient Boosting's 95.79%.
- **Recall:** Gradient Boosting performed better, detecting 96% of actual failures compared with the neural network's 93.20%.
- **Precision and F1 Score:** The neural network performed better on both metrics, meaning its positive predictions were more reliable overall.
- **Total Cost:** Gradient Boosting achieved the lower reported cost of 11,220, compared with the neural network's 12,150.
- **Average Precision:** The neural network achieved a slightly higher average precision across the three seeds (0.8220 vs. 0.8114).

**Note:** These results are from validation experiments used for model comparison and threshold selection. They should not be presented as final test-set results. The separate test dataset should only be used for the final evaluation.

## 3. Gradient Boosting Evaluation: Final Test Results

Final results of `model.gradient_eval` for the Gradient Boosting Classifier, evaluated on the **held-out test set**.

Run with:

```bash
py -m model.gradient_eval
```

### Final Test Set Results

| Metric    | Value      |
|-----------|------------|
| Accuracy  | 0.9606     |
| Recall    | 0.9680     |
| Precision | 0.3697     |
| F1        | 0.5350     |
| Total Cost| 12,190.00  |

The model catches almost all positive cases (96.8% recall). Precision is low (36.97%) because the positive class is very rare, so a number of false alarms is the trade-off for high recall.

### Dataset used in this evaluation

- Full dataset: 60,000 samples, 170 features
- Train/validation split: 45,000 / 15,000
- Positive class rate in the 15,000-sample evaluation split: 250 positives (about 1.67%)

### Trivial baseline (always predicts negative)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9833 |
| Recall    | 0.0000 |
| Precision | 0.0000 |
| F1        | 0.0000 |

Accuracy is misleading here: the trivial classifier scores 98.33% while catching zero positives.

### Threshold sweep (cost-based selection)

Thresholds from 0.0001 to 0.1 were swept, and the one with the lowest total cost was chosen.

- **Best threshold: 0.005** (total cost 11,220.00)
- Confusion matrix at the best threshold:

|                 | Predicted Neg | Predicted Pos |
|-----------------|---------------|---------------|
| **Actual Neg**  | 14,128        | 622           |
| **Actual Pos**  | 10            | 240           |

The cost values are consistent with a cost of 10 per false positive and 500 per false negative.

Selected points from the sweep:

| Threshold | Accuracy | Recall | Precision | F1     | Total Cost |
|-----------|----------|--------|-----------|--------|------------|
| 0.001     | 0.0167   | 1.0000 | 0.0167    | 0.0328 | 147,500    |
| 0.002     | 0.9173   | 0.9760 | 0.1651    | 0.2824 | 15,340     |
| **0.005** | 0.9579   | 0.9600 | 0.2784    | 0.4317 | **11,220** |
| 0.01      | 0.9673   | 0.9480 | 0.3319    | 0.4917 | 11,270     |
| 0.05      | 0.9803   | 0.9160 | 0.4544    | 0.6074 | 13,250     |
| 0.1       | 0.9863   | 0.8880 | 0.5564    | 0.6841 | 15,770     |

### Robustness: Average Precision Across Seeds (Gradient Boosting)

| Seed | Average Precision |
|------|-------------------|
| 42   | 0.8144            |
| 67   | 0.8107            |
| 69   | 0.8092            |
| **Mean** | **0.8114**    |

Scores are stable across seeds.

### Top Feature Importances

| Rank | Feature  | Importance |
|------|----------|------------|
| 1    | `bj_000` | 0.1585     |
| 2    | `ci_000` | 0.0785     |
| 3    | `ck_000` | 0.0741     |
| 4    | `ag_001` | 0.0478     |
| 5    | `ee_007` | 0.0383     |
| 6    | `ay_005` | 0.0372     |
| 7    | `ag_002` | 0.0320     |
| 8    | `ay_006` | 0.0296     |
| 9    | `am_0`   | 0.0285     |
| 10   | `ay_002` | 0.0243     |

Several of the top features (`bj_000`, `ci_000`, `ck_000`, `am_0`) are among those with missing values, so imputation choices may matter.

## 4. Imbalance Handling Experiments

At least two imbalance-handling approaches will be compared.

Experiments will investigate how different approaches affect:

- Recall
- Precision
- F1 score
- Average precision
- Total classification cost

| Approach | Description | Results |
|---|---|---|
| Baseline | Original class distribution | TBD |
| Approach 2 | TBD | TBD |

The final approach will be selected based on validation performance rather than test-set performance.

## 5. Error Analysis

Model evaluation should go beyond reporting a single metric.

The final analysis will investigate:

- False negatives: Actual APS failures that the model incorrectly predicts as normal.
- False positives: Normal operating conditions incorrectly classified as failures.
- Whether certain features or missing-value patterns are associated with prediction errors.
- Whether the selected classification threshold produces an acceptable balance between missed failures and unnecessary inspections.

The purpose is to understand where the model fails and whether those failures reveal weaknesses in the data, preprocessing, or model architecture.

**Results: To be completed using the final model predictions.**

---

# Part B: Method and Setup

## 6. Dataset

**Dataset:** APS Failure and Operational Data for Scania Trucks

The dataset contains operational information collected from heavy Scania trucks.

- Training dataset: `aps_failure_training_set.csv`
- Test dataset: `aps_failure_test_set.csv`
- Input features: 170
- Target column: `class`
- Target classes:
  - `neg`: No APS failure
  - `pos`: APS failure

The original CSV files contain metadata and license information before the actual column headers. These lines are skipped when loading the dataset.

### Class imbalance

The test dataset contains:

| Class | Samples |
|---|---:|
| Negative (`neg`) | 15,625 |
| Positive (`pos`) | 375 |
| Total | 16,000 |

This means that only approximately 2.34% of the test samples represent actual failures.

Because of this imbalance, accuracy alone is not sufficient to evaluate model performance.

## 7. Data Preparation

### Dataset splitting

The provided training dataset is divided into training and validation sets using a stratified split.

Current configuration:

- Training: 75% (45,000 samples)
- Validation: 25% (15,000 samples)
- Random state: 42

The provided test dataset remains separate and is reserved for final evaluation.

This produces an approximately 60/20/20 overall training, validation, and test distribution.

Stratification is used to preserve the proportion of positive and negative samples across the training and validation sets.

### Missing values

The APS dataset contains a substantial number of missing values.

Some features have particularly high missing-value percentages, including:

- `ab_000`
- `br_000`
- `bq_000`
- `bp_000`
- `bo_000`

Median imputation was selected as the initial missing-value handling strategy. since if values differ in a big amount mean imputation will cause a data bias.

The imputer is fitted using training data only. The same fitted imputer is then used to transform validation data.

The test dataset must also use the previously fitted imputer rather than fitting a new one.

Missing-value indicators were also explored to investigate whether the absence of certain sensor measurements provides useful information about potential failures.

### Standardization

Feature standardization is performed using statistics calculated from the training dataset.

The training mean and standard deviation are reused when transforming validation and test data.

This prevents information from the validation or test datasets from influencing the fitted preprocessing parameters.

## 8. Neural Network Architecture

The primary model is a feedforward neural network implemented using PyTorch.

### Architecture

| Layer | Configuration |
|---|---|
| Input layer | 170 features |
| Hidden layer | 64 neurons |
| Activation | ReLU |
| Output layer | 2 neurons |
| Loss function | CrossEntropyLoss |
| Optimizer | SGD |
| Learning rate | 0.01 |

The network follows this structure:

Input (170) → Linear (64) → ReLU → Linear (2)

The output layer produces two logits representing the negative and positive classes.

### Training configuration

| Parameter | Value |
|---|---:|
| Maximum epochs | 120 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Optimizer | SGD |
| Early stopping patience | 20 |
| Random seed | 42 |

Training and validation losses are monitored during training.

Early stopping is used to stop training when validation loss fails to improve for the specified patience period.

The best model is saved when validation loss improves, rather than automatically keeping the model from the final epoch.

## 9. Checkpointing and Resuming Training

The training process supports checkpoint saving and resuming.

Two checkpoint files are maintained:

- `best_model_{random_state}.pth`: Stores the model with the best validation loss.
- `latest_model_{random_state}.pth`: Stores the latest training state.

The checkpoint contains:

- Model state dictionary
- Optimizer state dictionary
- Current epoch
- Best epoch
- Best validation loss
- Number of epochs without improvement

When training is resumed, the latest checkpoint is loaded so that training can continue from the previously completed epoch.

A pytest test is being developed to verify that training resumes from the saved epoch and continues rather than restarting from epoch zero.

**Remaining verification:** Compare an uninterrupted training run against an interrupted-and-resumed run to establish that both produce equivalent results within a stated tolerance.

## 10. Baseline Models

The neural network is not evaluated in isolation.

Two baseline approaches are included in the project comparison.

### Trivial baseline

A simple baseline that predicts the majority class.

This provides a reference point for understanding how much performance a more sophisticated model actually adds.

### Gradient Boosting Classifier

A gradient-boosted tree classifier is used as another baseline.

This allows comparison between a traditional tree-based machine learning approach and the PyTorch neural network.

Both approaches should use the same training and validation split.

The final comparison is based on validation results during model development and test results for the final evaluation.

## 11. Evaluation Metrics and Error Costs

Accuracy is not the primary evaluation metric because of the severe class imbalance.

The following metrics are used:

- **Recall:** Measures how many actual failures the model successfully identifies.
- **Precision:** Measures how many predicted failures are actually failures.
- **F1 Score:** Balances precision and recall.
- **Average Precision:** Evaluates how well the model ranks positive samples above negative samples across different classification thresholds.
- **Accuracy:** Measures overall classification correctness.

### Cost-sensitive evaluation

False negatives and false positives do not have the same consequences.

For this project, the following cost assumptions are used:

| Error | Cost |
|---|---:|
| False Positive | 10 |
| False Negative | 500 |

These costs are project assumptions, not values supplied by the dataset.

A false negative is assigned a cost 50 times greater than a false positive because failing to identify a potential truck failure is considered substantially more expensive than unnecessarily flagging a truck for inspection.

The total classification cost is calculated as:

`Total Cost = (False Positives × 10) + (False Negatives × 500)`

The classification threshold is also investigated using validation data.

The final threshold should minimize validation classification cost while maintaining an acceptable ability to identify failures.

## 12. Leakage Audit

Preventing data leakage is important because it can produce misleadingly high evaluation results.

The following precautions are used:

1. The test dataset is kept separate from training and validation.
2. The imputer is fitted using training data only.
3. Standardization statistics are calculated using training data only.
4. Validation data is used for model selection and threshold selection.
5. The test dataset is reserved for final evaluation.
6. Preprocessing parameters are saved and reused during evaluation.

An additional feature-level investigation is needed to determine whether any operational columns contain information that would not actually be available when making a real-world prediction.

**Remaining work:** Document the features investigated, their potential leakage risks, and the decisions made about retaining or removing them.

## 13. Experiment Tracking

Experiments are recorded to track model development and make comparisons reproducible.

Each experiment should record:

- Experiment name
- Random seed
- Model configuration
- Changes made
- Reason for the changes
- Validation results
- Test results, when applicable
- Observations and conclusions

The experiment log should include unsuccessful experiments as well as successful ones.

### Experiment log

| Experiment | Description | Status |
|---|---|---|
| Initial data exploration | Investigated dataset structure, class distribution and missing values | Completed |
| Median imputation | Tested median imputation for missing features | Implemented |
| Missing-value indicators | Investigated missingness indicators | Explored |
| Neural network | Implemented a 170 → 64 → 2 architecture | Implemented |
| Gradient Boosting | Developed a gradient-boosted tree baseline | Implemented |
| Threshold optimization | Investigated classification thresholds using validation data | Implemented |
| Checkpoint resuming | Implemented latest and best checkpoint saving | Implemented |
| Resume pytest | Testing interrupted and resumed training | Done |
| Imbalance comparison | Comparing at least two imbalance-handling approaches | Pending |
| Multi-seed comparison | Comparing results across at least three seeds | Done |
| Error analysis | Investigating false positives and false negatives | Pending |

## 14. Reproducibility and Saved Artifacts

### Running the project

Run the commands from the project root directory (Pytorch-week-2), not from inside the model/ folder.

First, activate your virtual environment and install the required dependencies.

Then run the following commands:

1. PyTorch model evaluation/training

```bash
py -m model.train_eval
```

2. Gradient Boosting model evaluation

```bash
py -m model.gradient_eval
```

The project saves trained models and preprocessing objects for later evaluation.

Current artifacts include:

- `gradient_boost_classifier.pkl`
- `gradient_boost_threshold.pkl`
- `impute_and_standardize.pkl`
- `best_model_{random_state}.pth`
- `latest_model_{random_state}.pth`

The gradient boosting evaluation process loads the saved classifier, threshold, imputer, and standardization parameters rather than fitting them again.

The final PyTorch checkpoint should also contain the preprocessing information required to reproduce predictions independently of the training process.

**Remaining work:**
- Finalize the reproducible evaluation entry point.
- Publish the selected trained checkpoint as a release asset.
- Verify that the README's reported results can be reproduced using the saved artifacts.
- Complete the multi-seed comparison.

## 15. Project Structure

Current project organization:

```text
Pytorch-week-2/
│
├── .github/workflows
│     └── ci.yml
│
├── data/
│   ├── aps_failure_training_set.csv
│   └── aps_failure_test_set.csv
│
├── model/
│   ├── __init__.py
│   ├── data.py
│   ├── dataloader.py
│   └── classifier.py
│   ├── gradient_eval.py
│   ├──train_eval.py
│   ├──train_test.py
│   ├──train.py
│
├── tests/
│   └── test_train.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

Additional files, such as experiment logs, saved checkpoints, and evaluation scripts, are maintained as the project develops.

## 16. AI Assistance

ChatGPT was used as a learning and debugging assistant throughout this project.

Assistance included:

- Explaining PyTorch concepts and model training.
- Explaining what joblib does
- Explaining Accuracy, Recall, Precision,and F1
- Debugging errors in data loading and preprocessing.
- Explaining checkpoint saving and resuming logic.
- Explaining class imbalance and evaluation metrics.
- Helping understand pytest and test failures.
- Debugging implementation decisions and suggesting areas to investigate and telling the cause of error.
- Formatting and help on README.md
- Boilerplate for the plot 
- Boilerplate for the ci.yml

---

# Part C: Problems and Conclusion

## 17. Problems Encountered (Failed Runs)

```
  File "C:\Users\AD\Downloads\Pytorch-week-2\model\train.py", line 124, in <module>
    plot_losses(train_losses, val_losses, best_epoch, save_path="loss_curve.png")
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\AD\Downloads\Pytorch-week-2\model\train.py", line 111, in plot_losses
    plt.scatter(best_epoch + 1, val_losses[best_epoch], color="red", zorder=5)
                                ~~~~~~~~~~^^^^^^^^^^^^
IndexError: list index out of range
```

this error occured when I was refactoring and fixing my checkpointing. since the last committed checkpoint it only allows it to save the best validation loss not the model itself. The following errors occured but is not shown in the run: 

Upon refactoring, the early stopping broke down where it doesnt save the val_losses and average_train_loss because of a logic error and indentation error where it is coded as 

```
else: epochs_no_improve >= patience:
                epochs_no_improve += 1
                print(f"Early stopping at epoch {epoch+1}")
                break
```

since else statements doesnt allow conditions i have separated it into two. else will only add the epoch no improve and the patience will indented the same as torch.no_grad() and after the else statement since the current if statement contains "break" which stops the loop.

```
Traceback (most recent call last):
  File "C:\Users\AD\Downloads\Pytorch-week-2\model\classifier.py", line 118, in <module>
    mean_ap = average_precision_at_three_seeds()
  File "C:\Users\AD\Downloads\Pytorch-week-2\model\classifier.py", line 108, in average_precision_at_three_seeds
    classifier, _, gradient_probabilities = gradient_boost_classifier(seed=random_state)
                                            ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
TypeError: gradient_boost_classifier() got an unexpected keyword argument 'seed'
```

the error is fixed when I removed the seed=random_state and simply used random_state since it is already declared in the function.

```
Traceback (most recent call last):
  File "C:\Users\AD\Downloads\Pytorch-week-2\model\train.py", line 135, in <module>
    model, threshold_results, model_probabilities,y_val = train_threshold_classifier()
                                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\AD\Downloads\Pytorch-week-2\model\train.py", line 111, in train_threshold_classifier
    x_train, x_val, y_train, y_val = preprocess_data(x_train, x_val, y_train, y_val)
                                                     ^^^^^^^
UnboundLocalError: cannot access local variable 'x_train' where it is not associated with a value
```

Fix: removed the unused variable and replaced it with _ 

Latest commmit for train.py. 
the model kept training for three times even if a single command is used. To fix that I refactored the training eval into a separate file. through the train_eval.py

Error Analysis: The model is currently overwriting the best result and when ran again it gets back to Epoch 0. first assumption is that all seed is stored at best_model.pth and used variables are local variables.

fix: the fix was removing stray calls and splitting into two seed files and separating the three seeds with different names, for the overwriting part the variables are set as attributes in the checkpoint instead of being a local variable which resets to epoch 0 always

## 18. Conclusion

The PyTorch neural network achieved better accuracy, precision, F1 score, and average precision across three seeds. However, Gradient Boosting achieved the lowest reported total cost and higher recall.

Since the project assigns a higher penalty to false negatives, Gradient Boosting is the better choice based on the lowest reported total cost. The neural network remains competitive and performs better on several other evaluation metrics.

The final conclusion will compare the PyTorch neural network against the trivial baseline and gradient-boosted tree classifier.

The objective is not necessarily to achieve the highest accuracy or outperform every baseline. It is to determine whether the neural network provides a reliable and practically useful approach to APS failure prediction.