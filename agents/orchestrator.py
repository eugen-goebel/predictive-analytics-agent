"""
Orchestrator — Coordinates the 6-phase ML pipeline.

Pipeline:
  DataProfiler → Preprocessor → FeatureEngineer → ModelTrainer → Evaluator → ReportGenerator
"""

import os
import tempfile

from agents.data_profiler import DataProfiler
from agents.preprocessor import PreprocessorAgent
from agents.feature_engineer import FeatureEngineerAgent
from agents.model_trainer import ModelTrainerAgent
from agents.evaluator import EvaluatorAgent
from utils.report_generator import generate_docx_report


class MLPipelineOrchestrator:
    """
    Coordinates the full machine learning pipeline from raw data to report.
    No API calls required — runs entirely locally.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir

    def run(self, filepath: str, tune: bool = False) -> str:
        """
        Execute the full 6-phase ML pipeline.

        Args:
            filepath: Path to CSV or Excel file
            tune:     Run hyperparameter tuning with GridSearchCV

        Returns:
            Absolute path to the generated DOCX report
        """
        # Phase 1: Profile
        print(f"\n[1/6] Profiling dataset ...")
        profiler = DataProfiler()
        profile, df = profiler.profile(filepath)
        print(f"      {profile.row_count:,} rows, {profile.column_count} columns")
        print(f"      Target: '{profile.target_column}' ({profile.task_type})")
        print(f"      Quality: {profile.data_quality_score}/100")

        # Phase 2: Preprocess
        print(f"\n[2/6] Preprocessing data ...")
        preprocessor = PreprocessorAgent()
        preprocess_result, X, y = preprocessor.preprocess(df, profile)
        print(f"      {preprocess_result.n_samples} samples, {preprocess_result.n_features} features")
        print(f"      Steps: {len(preprocess_result.steps_applied)}")

        # Phase 3: Feature selection
        print(f"\n[3/6] Selecting best features ...")
        engineer = FeatureEngineerAgent()
        feature_result, X_selected = engineer.select_features(
            X, y, preprocess_result.feature_names, profile.task_type
        )
        print(f"      Selected {feature_result.selected_feature_count} of "
              f"{feature_result.original_feature_count} features")
        print(f"      Method: {feature_result.method}")

        # Phase 4: Train models
        if tune:
            print(f"\n[4/6] Training models with hyperparameter tuning ...")
        else:
            print(f"\n[4/6] Training models ...")
        trainer = ModelTrainerAgent()
        training_result = trainer.train(X_selected, y, profile.task_type, tune=tune)
        print(f"      Best: {training_result.best_model_name} "
              f"(score: {training_result.best_score:.4f})")
        print(f"      Time: {training_result.training_time_seconds:.1f}s")

        # Phase 5: Evaluate
        chart_dir = os.path.join(self.output_dir, "charts")
        print(f"\n[5/6] Evaluating best model ...")
        evaluator = EvaluatorAgent()
        eval_result = evaluator.evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=feature_result.selected_features,
            task_type=profile.task_type,
            output_dir=chart_dir,
        )
        print(f"      Test score: {eval_result.test_score:.4f}")
        print(f"      Overfitting: {'Yes' if eval_result.is_overfitting else 'No'}")
        print(f"      Charts: {len(eval_result.charts)}")

        # Phase 6: Report
        print(f"\n[6/6] Generating DOCX report ...")
        report_path = generate_docx_report(
            profile=profile,
            preprocess_result=preprocess_result,
            feature_result=feature_result,
            training_result=training_result,
            eval_result=eval_result,
            output_dir=self.output_dir,
        )
        print(f"      Report: {report_path}")

        return report_path
