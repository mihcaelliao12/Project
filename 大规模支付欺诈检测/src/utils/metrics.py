from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
    )
import pandas as pd


# 根据给定阈值对概率预测打分并生成报告，返回 y_pred。
def evaluate_threshold(y_test, y_proba, threshold=0.5, show_matrix=True):
    
     # 根据阈值转为 0/1
    y_pred = (y_proba >= threshold).astype(int)

    # 计算主要指标
    metrics = {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred)*100,
        "Precision": precision_score(y_test, y_pred, zero_division=0)*100,
        "Recall": recall_score(y_test, y_pred, zero_division=0)*100,
        "F1": f1_score(y_test, y_pred, zero_division=0)*100,
        "ROC-AUC": roc_auc_score(y_test, y_proba)*100,
        "PR-AUC": average_precision_score(y_test, y_proba)*100
    }

    print("\n📊 Model Evaluation at Threshold =", round(threshold, 3))
    display(pd.DataFrame(metrics, index=["Score"]).T)

    if show_matrix:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))
    
    return y_pred


# 和其他预测结果比较分数
def compare_models(y_true, new_pred, old_pred):
    metrics = {}
    for name, pred in [('New model', new_pred), ('Old Model', old_pred)]:
        metrics[name] = {
            'Precision': precision_score(y_true, pred),
            'Recall': recall_score(y_true, pred),
            'F1': f1_score(y_true, pred),
            'Accuracy': accuracy_score(y_true, pred)
        }
    return pd.DataFrame(metrics)