from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "loan_default"


@dataclass(slots=True)
class DatasetBundle:
    feature_names: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])


def generate_credit_risk_dataframe(
    num_rows: int = 100_000,
    total_features: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    if total_features < 20:
        raise ValueError("total_features must be at least 20 to include the core credit-risk signals")

    rng = np.random.default_rng(seed)

    customer_age = rng.integers(21, 71, size=num_rows)
    annual_income = np.clip(rng.lognormal(mean=11.0, sigma=0.45, size=num_rows), 20_000, 260_000)
    loan_amount = np.clip(annual_income * rng.uniform(0.08, 0.65, size=num_rows), 5_000, 120_000)
    credit_score = np.clip(rng.normal(loc=680, scale=85, size=num_rows), 300, 850)
    employment_years = np.clip(customer_age - rng.integers(18, 28, size=num_rows), 0, 42)
    existing_loans = rng.poisson(lam=1.6, size=num_rows)
    late_payment_count = rng.poisson(lam=1.1, size=num_rows)
    debt_to_income_ratio = np.clip(rng.beta(a=2.4, b=3.2, size=num_rows), 0.03, 0.85)
    loan_tenure = rng.integers(12, 85, size=num_rows)
    interest_rate = np.clip(rng.normal(loc=0.12, scale=0.05, size=num_rows), 0.03, 0.34)

    loan_to_income_ratio = np.clip(loan_amount / np.maximum(annual_income, 1.0), 0.03, 1.25)
    credit_utilization_ratio = np.clip(rng.beta(a=2.0, b=2.2, size=num_rows), 0.02, 0.99)
    recent_credit_inquiries = rng.poisson(lam=1.8, size=num_rows)
    payment_to_income_ratio = np.clip(
        (loan_amount / np.maximum(loan_tenure, 1.0)) / np.maximum(annual_income / 12.0, 1.0),
        0.01,
        1.5,
    )
    income_stability_score = np.clip(rng.normal(loc=0.72, scale=0.16, size=num_rows), 0.0, 1.0)
    savings_buffer_months = np.clip(rng.normal(loc=4.5, scale=2.3, size=num_rows), 0.0, 18.0)
    open_credit_lines = np.clip(rng.poisson(lam=5.5, size=num_rows), 1, 18)
    credit_history_years = np.clip(customer_age - 18 - rng.integers(0, 8, size=num_rows), 1, 45)
    spending_volatility = np.clip(rng.normal(loc=0.42, scale=0.18, size=num_rows), 0.0, 1.0)
    emergency_fund_ratio = np.clip(rng.lognormal(mean=-0.4, sigma=0.55, size=num_rows), 0.05, 3.5)

    base_features: dict[str, np.ndarray] = {
        "customer_age": customer_age.astype(np.float32),
        "annual_income": annual_income.astype(np.float32),
        "loan_amount": loan_amount.astype(np.float32),
        "credit_score": credit_score.astype(np.float32),
        "employment_years": employment_years.astype(np.float32),
        "existing_loans": existing_loans.astype(np.float32),
        "late_payment_count": late_payment_count.astype(np.float32),
        "debt_to_income_ratio": debt_to_income_ratio.astype(np.float32),
        "loan_tenure": loan_tenure.astype(np.float32),
        "interest_rate": interest_rate.astype(np.float32),
        "loan_to_income_ratio": loan_to_income_ratio.astype(np.float32),
        "credit_utilization_ratio": credit_utilization_ratio.astype(np.float32),
        "recent_credit_inquiries": recent_credit_inquiries.astype(np.float32),
        "payment_to_income_ratio": payment_to_income_ratio.astype(np.float32),
        "income_stability_score": income_stability_score.astype(np.float32),
        "savings_buffer_months": savings_buffer_months.astype(np.float32),
        "open_credit_lines": open_credit_lines.astype(np.float32),
        "credit_history_years": credit_history_years.astype(np.float32),
        "spending_volatility": spending_volatility.astype(np.float32),
        "emergency_fund_ratio": emergency_fund_ratio.astype(np.float32),
    }

    remaining_features = total_features - len(base_features)
    signal_pool = np.column_stack(
        [
            loan_to_income_ratio,
            debt_to_income_ratio,
            credit_utilization_ratio,
            interest_rate,
            late_payment_count,
            existing_loans,
            spending_volatility,
            emergency_fund_ratio,
            income_stability_score,
        ]
    )
    for index in range(remaining_features):
        weights = rng.normal(loc=0.0, scale=0.55, size=signal_pool.shape[1])
        generated_feature = signal_pool @ weights + rng.normal(loc=0.0, scale=0.75, size=num_rows)
        feature_name = f"generated_feature_{index + 1:02d}"
        base_features[feature_name] = generated_feature.astype(np.float32)

    risk_score = (
        1.8 * loan_to_income_ratio
        + 2.2 * debt_to_income_ratio
        + 1.7 * credit_utilization_ratio
        + 0.28 * late_payment_count
        + 0.18 * existing_loans
        + 0.065 * recent_credit_inquiries
        + 3.5 * interest_rate
        + 0.75 * spending_volatility
        - 0.010 * (credit_score - 300)
        - 0.045 * employment_years
        - 0.065 * savings_buffer_months
        - 0.90 * income_stability_score
        - 0.55 * emergency_fund_ratio
        + rng.normal(loc=0.0, scale=0.55, size=num_rows)
        - 1.2
    )
    default_probability = 1.0 / (1.0 + np.exp(-risk_score))
    target = rng.binomial(n=1, p=np.clip(default_probability, 0.02, 0.97), size=num_rows)

    dataframe = pd.DataFrame(base_features)
    dataframe[TARGET_COLUMN] = target.astype(np.int8)
    return dataframe


def split_credit_risk_dataframe(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    feature_names = [column for column in dataframe.columns if column != TARGET_COLUMN]
    X = dataframe[feature_names].to_numpy(dtype=np.float32)
    y = dataframe[TARGET_COLUMN].to_numpy(dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetBundle(
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def load_credit_risk_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def load_or_generate_bundle(
    dataset_path: str | Path | None = None,
    num_rows: int = 100_000,
    total_features: int = 50,
    seed: int = 42,
) -> DatasetBundle:
    if dataset_path:
        dataframe = load_credit_risk_dataframe(dataset_path)
    else:
        dataframe = generate_credit_risk_dataframe(
            num_rows=num_rows,
            total_features=total_features,
            seed=seed,
        )
    return split_credit_risk_dataframe(dataframe, random_state=seed)


def save_credit_risk_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return destination


def summarize_dataframe(dataframe: pd.DataFrame) -> dict[str, float | int]:
    default_rate = float(dataframe[TARGET_COLUMN].mean())
    return {
        "rows": int(len(dataframe)),
        "feature_count": int(len(dataframe.columns) - 1),
        "default_rate": default_rate,
    }
