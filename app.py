"""
Predictive Analytics Agent — Streamlit Web Interface.

Upload a CSV/Excel file and the pipeline automatically:
  1. Profiles the dataset
  2. Preprocesses and selects features
  3. Trains & compares 4 ML models
  4. Shows evaluation results with charts

No API key required — everything runs locally.

To run:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
import pandas as pd

from agents.data_profiler import DataProfiler
from agents.preprocessor import PreprocessorAgent
from agents.feature_engineer import FeatureEngineerAgent
from agents.model_trainer import ModelTrainerAgent
from agents.evaluator import EvaluatorAgent
from utils.report_generator import generate_docx_report


st.set_page_config(
    page_title="Predictive Analytics Agent",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 ML Pipeline")
    st.info("**No API key needed** — runs entirely on your machine using scikit-learn.")
    st.divider()

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file with a target column",
    )

    # Load sample button
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_customers.csv")
    use_sample = False
    if os.path.exists(sample_path):
        if st.button("📋 Use sample dataset (Customer Churn)"):
            use_sample = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Predictive Analytics Agent")
st.caption("Upload a dataset and the ML pipeline automatically trains, compares, and evaluates models.")

# Determine file path
filepath = None
if use_sample:
    filepath = sample_path
elif uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        filepath = tmp.name

if filepath:
    try:
        with st.spinner("Phase 1/6 — Profiling dataset..."):
            profiler = DataProfiler()
            profile, df = profiler.profile(filepath)

        # --- Data Profile ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{profile.row_count:,}")
        col2.metric("Columns", str(profile.column_count))
        col3.metric("Task", profile.task_type.capitalize())
        col4.metric("Quality", f"{profile.data_quality_score}%")

        st.subheader(f"Target: `{profile.target_column}` ({profile.task_type})")

        # Show data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        # Class distribution
        if profile.class_distribution:
            with st.expander("📈 Class Distribution"):
                st.bar_chart(pd.Series(profile.class_distribution))

        st.divider()

        # --- Preprocess ---
        with st.spinner("Phase 2/6 — Preprocessing..."):
            preprocessor = PreprocessorAgent()
            preprocess_result, X, y = preprocessor.preprocess(df, profile)

        with st.expander("🔧 Preprocessing Steps", expanded=False):
            for step in preprocess_result.steps_applied:
                st.write(f"• {step}")

        # --- Feature Selection ---
        with st.spinner("Phase 3/6 — Selecting features..."):
            engineer = FeatureEngineerAgent()
            feature_result, X_selected = engineer.select_features(
                X, y, preprocess_result.feature_names, profile.task_type
            )

        st.write(
            f"**Features:** {feature_result.selected_feature_count} selected "
            f"from {feature_result.original_feature_count} "
            f"({feature_result.method})"
        )

        # --- Train ---
        with st.spinner("Phase 4/6 — Training 4 models..."):
            trainer = ModelTrainerAgent()
            training_result = trainer.train(X_selected, y, profile.task_type)

        st.divider()
        st.subheader("Model Comparison")

        # Model comparison table
        metric_name = "Accuracy" if profile.task_type == "classification" else "R² Score"
        model_data = {
            "Model": [m.name for m in training_result.model_scores],
            metric_name: [f"{m.score:.4f}" for m in training_result.model_scores],
            "Std Dev": [f"±{m.cv_std:.4f}" for m in training_result.model_scores],
            "Time": [f"{m.training_time:.3f}s" for m in training_result.model_scores],
        }
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

        st.success(
            f"**Best Model: {training_result.best_model_name}** "
            f"({metric_name}: {training_result.best_score:.4f})"
        )

        # --- Evaluate ---
        chart_dir = tempfile.mkdtemp(prefix="ml_charts_")
        with st.spinner("Phase 5/6 — Evaluating..."):
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

        st.divider()
        st.subheader("Evaluation Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Score", f"{eval_result.test_score:.4f}")
        col2.metric("Train Score", f"{eval_result.train_score:.4f}")
        col3.metric("Overfitting", "Yes ⚠️" if eval_result.is_overfitting else "No ✓")

        # Charts
        for chart in eval_result.charts:
            chart_path = os.path.join(chart_dir, chart.filename)
            if os.path.exists(chart_path):
                st.image(chart_path, caption=chart.description, use_container_width=True)

        # Classification report
        if eval_result.classification_report:
            with st.expander("📋 Classification Report"):
                st.code(eval_result.classification_report)

        # --- Report ---
        st.divider()
        output_dir = tempfile.mkdtemp(prefix="ml_report_")
        # Copy charts to output dir
        import shutil
        report_chart_dir = os.path.join(output_dir, "charts")
        shutil.copytree(chart_dir, report_chart_dir)

        with st.spinner("Phase 6/6 — Generating report..."):
            report_path = generate_docx_report(
                profile=profile,
                preprocess_result=preprocess_result,
                feature_result=feature_result,
                training_result=training_result,
                eval_result=eval_result,
                output_dir=output_dir,
            )

        with open(report_path, "rb") as f:
            st.download_button(
                "📥 Download Full Report (DOCX)",
                data=f.read(),
                file_name=os.path.basename(report_path),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        # Clean up uploaded temp file
        if uploaded and filepath and os.path.exists(filepath):
            os.unlink(filepath)

else:
    st.info("Upload a CSV/Excel file or load the sample dataset to start the ML pipeline.")
