from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Splits data into training and testing sets using sklearn's train_test_split.
    """
    logger.info(f"Splitting data → test size: {test_size}, stratified: {stratify is not None}")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def evaluate_model(y_true, y_pred, y_proba, model_name="Model", save_roc_path=None):
    """
    Computes and prints standard classification metrics and plots ROC curve.
    Saves ROC plot if save_roc_path is provided.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rocauc = roc_auc_score(y_true, y_proba)

    logger.info(f"Evaluating model: {model_name}")
    print(f"\n📌 ====== {model_name} Evaluation ======")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {rocauc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")

    if save_roc_path:
        plt.savefig(save_roc_path)
        logger.info(f"ROC curve saved to: {save_roc_path}")
    else:
        plt.show()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": rocauc,
    }


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Displays or saves confusion matrix plot.
    """
    logger.info(f"Plotting confusion matrix: {model_name}")
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Non-Fraud", "Fraud"], cmap="Blues", values_format="d"
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()