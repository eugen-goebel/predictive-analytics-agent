# Predictive Analytics Agent

An automated machine learning pipeline that profiles datasets, preprocesses data, selects features, trains and compares multiple models, and generates a professional evaluation report — all without requiring an API key.

![CI](https://github.com/eugen-goebel/predictive-analytics-agent/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Tests](https://img.shields.io/badge/Tests-35_passed-brightgreen)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5+-f7931e)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Auto-Detection**: Automatically identifies the target column and task type (classification or regression)
- **Data Profiling**: Analyzes data quality, distributions, missing values, and column statistics
- **Smart Preprocessing**: Handles missing values, encodes categoricals, and scales features
- **Feature Selection**: Applies variance thresholding and statistical feature selection (SelectKBest)
- **Model Comparison**: Trains 4 models with 5-fold cross-validation and selects the best
- **Evaluation**: Generates confusion matrices, feature importance charts, model comparison plots, and overfitting detection
- **Report Generation**: Creates a professional DOCX report with all results and visualizations
- **Web Interface**: Interactive Streamlit app for uploading data and exploring results
- **No API Key Required**: Runs entirely locally using scikit-learn

## Architecture

The project uses a multi-agent architecture where each agent handles one pipeline phase:

```
MLPipelineOrchestrator
├── DataProfiler          → Dataset analysis & target detection
├── PreprocessorAgent     → Cleaning, encoding, scaling
├── FeatureEngineerAgent  → Feature selection & ranking
├── ModelTrainerAgent     → Training & cross-validation (4 models)
├── EvaluatorAgent        → Metrics, charts, overfitting detection
└── ReportGenerator       → Professional DOCX report
```

### Models Used

**Classification:**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors

**Regression:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### CLI Usage

```bash
# Run with sample dataset
python main.py

# Run with your own data
python main.py path/to/your_data.csv

# Specify output directory
python main.py data.csv --output reports/
```

### Web Interface

```bash
streamlit run app.py
```

Upload a CSV/Excel file or use the built-in sample dataset. The app displays:
- Data profile with quality metrics
- Preprocessing steps applied
- Model comparison table with scores
- Evaluation charts (confusion matrix, feature importance, etc.)
- Download button for the full DOCX report

## Sample Dataset

Includes a customer churn dataset (`data/sample_customers.csv`) with 80 rows and 11 features:

| Feature | Description |
|---------|-------------|
| age | Customer age |
| income | Annual income |
| credit_score | Credit score |
| years_customer | Years as customer |
| num_products | Number of products |
| has_mortgage | Has mortgage (0/1) |
| has_online_banking | Uses online banking (0/1) |
| monthly_charges | Monthly charges |
| total_charges | Total charges |
| support_calls | Number of support calls |
| **churn** | **Target** — whether customer churned (0/1) |

## Testing

```bash
pytest tests/ -v
```

35 tests covering all agents and the end-to-end pipeline.

## Project Structure

```
predictive-analytics-agent/
├── agents/
│   ├── data_profiler.py        # Dataset profiling & analysis
│   ├── preprocessor.py         # Data cleaning & transformation
│   ├── feature_engineer.py     # Feature selection & ranking
│   ├── model_trainer.py        # Model training & comparison
│   ├── evaluator.py            # Model evaluation & charts
│   └── orchestrator.py         # Pipeline coordinator
├── utils/
│   └── report_generator.py     # DOCX report generation
├── tests/
│   ├── test_profiler.py
│   ├── test_preprocessor.py
│   ├── test_feature_engineer.py
│   ├── test_model_trainer.py
│   ├── test_evaluator.py
│   └── test_pipeline.py
├── data/
│   └── sample_customers.csv
├── app.py                      # Streamlit web interface
├── main.py                     # CLI entry point
└── requirements.txt
```

## Tech Stack

- **scikit-learn** — Machine learning models, preprocessing, evaluation
- **pandas** — Data manipulation
- **matplotlib** — Chart generation
- **Streamlit** — Web interface
- **python-docx** — Report generation
- **Pydantic** — Data validation with typed models

## License

MIT
