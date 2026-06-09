"""
Evaluator Agent — Tests the best model and creates visualization charts.

After the ModelTrainer picks the best model, this agent:
  1. Tests it on the holdout data (data the model has never seen)
  2. Checks for OVERFITTING (model memorized training data but fails on new data)
  3. Creates professional charts (confusion matrix, model comparison, etc.)

WHAT IS OVERFITTING?
  Imagine studying for an exam by memorizing the answers to practice questions.
  You score 100% on the practice (training), but if the real exam (test) has
  different questions, you fail. That's overfitting. We detect it when
  train_score >> test_score (difference > 10%).
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .model_trainer import TrainingResult

PERMUTATION_TOP_K = 15
PERMUTATION_N_REPEATS = 10


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChartInfo(BaseModel):
    """Metadata about a generated chart."""

    title: str
    filename: str
    description: str


class EvaluationResult(BaseModel):
    """Complete evaluation results."""

    test_score: float
    train_score: float
    is_overfitting: bool = Field(description="True if train-test gap > 10%")
    charts: list[ChartInfo]
    classification_report: str | None = None
    metrics: dict = Field(description="Additional metrics (precision, recall, MAE, etc.)")


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLORS = ["#0D7377", "#C44900", "#2E86AB", "#A23B72", "#F18F01"]
BG_COLOR = "#FAFAFA"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class EvaluatorAgent:
    """
    Evaluates the trained model on test data and generates charts.

    Usage:
        evaluator = EvaluatorAgent()
        result = evaluator.evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=["age", "income"],
            task_type="classification",
            output_dir="output/charts",
        )
    """

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        training_result: TrainingResult,
        feature_names: list[str],
        task_type: str,
        output_dir: str,
    ) -> EvaluationResult:
        """
        Evaluate the model and create charts.

        Args:
            model:           Fitted sklearn model
            X_test, y_test:  Holdout test data
            X_train, y_train: Training data (for overfitting check)
            training_result: TrainingResult from ModelTrainer
            feature_names:   Feature names for chart labels
            task_type:       "classification" or "regression"
            output_dir:      Directory to save chart PNGs

        Returns:
            EvaluationResult with metrics and chart metadata
        """
        os.makedirs(output_dir, exist_ok=True)

        # --- Scores ---
        if task_type == "classification":
            test_score = round(accuracy_score(y_test, model.predict(X_test)), 4)
            train_score = round(accuracy_score(y_train, model.predict(X_train)), 4)
        else:
            test_score = round(r2_score(y_test, model.predict(X_test)), 4)
            train_score = round(r2_score(y_train, model.predict(X_train)), 4)

        is_overfitting = (train_score - test_score) > 0.1

        # --- Metrics ---
        y_pred = model.predict(X_test)
        metrics: dict = {}
        cls_report = None

        if task_type == "classification":
            avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
            metrics["accuracy"] = test_score
            metrics["precision"] = round(
                float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4
            )
            metrics["recall"] = round(
                float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4
            )
            metrics["f1_score"] = round(
                float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4
            )
            cls_report = classification_report(y_test, y_pred, zero_division=0)
        else:
            metrics["r2_score"] = test_score
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

        # --- Charts ---
        charts: list[ChartInfo] = []

        # Chart 1: Model comparison
        chart = self._chart_model_comparison(training_result, task_type, output_dir)
        charts.append(chart)

        # Chart 2: Task-specific charts
        if task_type == "classification":
            chart = self._chart_confusion_matrix(y_test, y_pred, output_dir)
            charts.append(chart)
        else:
            chart = self._chart_actual_vs_predicted(y_test, y_pred, output_dir)
            charts.append(chart)
            chart = self._chart_residuals(y_test, y_pred, output_dir)
            charts.append(chart)

        # Chart 3: Native feature importance (only for tree-based models)
        if hasattr(model, "feature_importances_") and len(feature_names) > 0:
            chart = self._chart_feature_importance(model, feature_names, output_dir)
            if chart:
                charts.append(chart)

        # Chart 4: Permutation importance (model-agnostic, runs for every model)
        if len(feature_names) > 0:
            chart = self._chart_permutation_importance(
                model,
                X_test,
                y_test,
                feature_names,
                task_type,
                output_dir,
            )
            if chart:
                charts.append(chart)

        return EvaluationResult(
            test_score=test_score,
            train_score=train_score,
            is_overfitting=is_overfitting,
            charts=charts,
            classification_report=cls_report,
            metrics=metrics,
        )

    def _chart_model_comparison(
        self, result: TrainingResult, task_type: str, output_dir: str
    ) -> ChartInfo:
        """Bar chart comparing all models' CV scores."""
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        names = [m.name for m in result.model_scores]
        scores = [m.score for m in result.model_scores]
        stds = [m.cv_std for m in result.model_scores]

        bars = ax.barh(names, scores, xerr=stds, color=COLORS[: len(names)], height=0.5, capsize=5)

        metric_name = "Accuracy" if task_type == "classification" else "R² Score"
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_title(f"Model Comparison — {metric_name}", fontsize=14, fontweight="bold", pad=15)
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, score in zip(bars, scores, strict=False):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=10,
                color="#333",
            )

        plt.tight_layout()
        filename = "model_comparison.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return ChartInfo(
            title="Model Comparison",
            filename=filename,
            description=f"Cross-validation {metric_name.lower()} comparison across all models",
        )

    def _chart_confusion_matrix(self, y_true, y_pred, output_dir: str) -> ChartInfo:
        """Heatmap of the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor(BG_COLOR)

        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color=color,
                )

        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
        fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        filename = "confusion_matrix.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return ChartInfo(
            title="Confusion Matrix",
            filename=filename,
            description="Shows correct vs incorrect predictions for each class",
        )

    def _chart_actual_vs_predicted(self, y_true, y_pred, output_dir: str) -> ChartInfo:
        """Scatter plot of actual vs predicted values."""
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        ax.scatter(y_true, y_pred, alpha=0.6, color=COLORS[0], s=40)

        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color="#999", linewidth=1.5, label="Perfect prediction")

        ax.set_xlabel("Actual Values", fontsize=11)
        ax.set_ylabel("Predicted Values", fontsize=11)
        ax.set_title("Actual vs Predicted", fontsize=14, fontweight="bold", pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        filename = "actual_vs_predicted.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return ChartInfo(
            title="Actual vs Predicted",
            filename=filename,
            description="Points close to the diagonal line indicate accurate predictions",
        )

    def _chart_residuals(self, y_true, y_pred, output_dir: str) -> ChartInfo:
        """Residual plot (errors vs predicted values)."""
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        ax.scatter(y_pred, residuals, alpha=0.6, color=COLORS[1], s=40)
        ax.axhline(y=0, color="#999", linestyle="--", linewidth=1.5)

        ax.set_xlabel("Predicted Values", fontsize=11)
        ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
        ax.set_title("Residual Plot", fontsize=14, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        filename = "residuals.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return ChartInfo(
            title="Residual Plot",
            filename=filename,
            description="Residuals should be randomly scattered around zero",
        )

    def _chart_permutation_importance(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        task_type: str,
        output_dir: str,
    ) -> ChartInfo | None:
        """Model-agnostic permutation importance — works for any fitted estimator.

        For each feature, sklearn shuffles its column n_repeats times and
        measures the drop in the chosen score. Larger drop = more important.
        Unlike ``feature_importances_`` this is defined for every model,
        including linear models and KNN.
        """
        try:
            scoring = "accuracy" if task_type == "classification" else "r2"
            n_features = min(len(feature_names), X_test.shape[1])
            if n_features == 0:
                return None

            result = permutation_importance(
                model,
                X_test[:, :n_features],
                y_test,
                n_repeats=PERMUTATION_N_REPEATS,
                random_state=42,
                scoring=scoring,
                n_jobs=1,
            )

            means = result.importances_mean
            names = feature_names[:n_features]

            # Top-K by absolute importance, then plot in ascending order
            top_k = min(PERMUTATION_TOP_K, n_features)
            top_idx = np.argsort(np.abs(means))[-top_k:]
            sorted_names = [names[i] for i in top_idx]
            sorted_means = means[top_idx]

            fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.4)))
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)

            ax.barh(sorted_names, sorted_means, color=COLORS[2], height=0.6)
            metric_label = "Accuracy drop" if task_type == "classification" else "R² drop"
            ax.set_xlabel(f"Permutation importance ({metric_label})", fontsize=11)
            ax.set_title(
                "Permutation Importance (model-agnostic)", fontsize=14, fontweight="bold", pad=15
            )
            ax.axvline(0, color="#888", linewidth=0.8)
            ax.grid(True, axis="x", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            filename = "permutation_importance.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
            plt.close(fig)

            return ChartInfo(
                title="Permutation Importance",
                filename=filename,
                description=(
                    "Drop in score when each feature is randomly shuffled — "
                    "works for any model (linear, tree, KNN)."
                ),
            )
        except Exception:
            return None

    def _chart_feature_importance(
        self, model, feature_names: list[str], output_dir: str
    ) -> ChartInfo | None:
        """Bar chart of feature importances (for tree-based models)."""
        try:
            importances = model.feature_importances_
            n = min(len(feature_names), len(importances))
            names = feature_names[:n]
            imps = importances[:n]

            sorted_idx = np.argsort(imps)
            sorted_names = [names[i] for i in sorted_idx]
            sorted_imps = imps[sorted_idx]

            fig, ax = plt.subplots(figsize=(10, max(4, n * 0.4)))
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)

            ax.barh(sorted_names, sorted_imps, color=COLORS[0], height=0.6)
            ax.set_xlabel("Importance", fontsize=11)
            ax.set_title("Feature Importance", fontsize=14, fontweight="bold", pad=15)
            ax.grid(True, axis="x", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            filename = "feature_importance.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
            plt.close(fig)

            return ChartInfo(
                title="Feature Importance",
                filename=filename,
                description="Shows which features have the most impact on predictions",
            )
        except Exception:
            return None
