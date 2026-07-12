"""Generate the bundled customer churn sample dataset.

The churn label is drawn probabilistically from a logistic score over the
features plus irreducible noise, so models reach realistic accuracy on the
demo instead of a suspicious 100%.

Run:
    python data/generate_sample_data.py
"""

import os

import numpy as np
import pandas as pd

N_CUSTOMERS = 400
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    age = rng.integers(18, 75, N_CUSTOMERS)
    income = rng.normal(58000, 18000, N_CUSTOMERS).clip(18000, 150000).round(-2).astype(int)
    credit_score = (
        (rng.normal(680, 55, N_CUSTOMERS) + (income - 58000) / 1500).clip(350, 850).astype(int)
    )
    years_customer = rng.integers(0, 20, N_CUSTOMERS)
    num_products = rng.integers(1, 6, N_CUSTOMERS)
    has_mortgage = rng.binomial(1, 0.4, N_CUSTOMERS)
    has_online_banking = rng.binomial(1, 0.7, N_CUSTOMERS)
    monthly_charges = (rng.normal(70, 25, N_CUSTOMERS) + num_products * 8).clip(15, 200).round(2)
    months_active = years_customer * 12 + rng.integers(1, 12, N_CUSTOMERS)
    total_charges = (monthly_charges * months_active * rng.normal(1.0, 0.08, N_CUSTOMERS)).round(2)
    support_calls = rng.poisson(1.6, N_CUSTOMERS).clip(0, 10)

    # Logistic churn score: frequent support callers, expensive plans and new
    # customers churn more, while tenure, product depth and online banking
    # retain. The noise term keeps the classes overlapping, so there is a
    # ceiling below 100% accuracy no matter the model.
    z = (
        0.80 * (support_calls - 1.6)
        + 0.030 * (monthly_charges - 70)
        - 0.22 * years_customer
        - 0.45 * (num_products - 2)
        - 0.50 * has_online_banking
        + rng.normal(0, 0.75, N_CUSTOMERS)
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churn = (rng.random(N_CUSTOMERS) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "years_customer": years_customer,
            "num_products": num_products,
            "has_mortgage": has_mortgage,
            "has_online_banking": has_online_banking,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "churn": churn,
        }
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_customers.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df)} rows, churn rate {churn.mean():.1%}")


if __name__ == "__main__":
    main()
