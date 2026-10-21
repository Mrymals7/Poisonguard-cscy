import numpy as np
from src.evaluation.metrics import evaluate, attack_success_rate


def test_evaluate_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    result = evaluate(y_true, y_pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["fpr"] == 0.0

def test_evaluate_one_false_alarm():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 1])
    result = evaluate(y_true, y_pred)
    assert result["recall"] == 1.0
    assert result["fpr"] == 0.5
    assert result["precision"] == 2 / 3


def test_attack_success_rate():
    y_pred_on_attacks = np.array([0, 1, 0, 1])
    assert attack_success_rate(y_pred_on_attacks) == 0.5