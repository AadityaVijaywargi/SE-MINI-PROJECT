"""Generate the bundled sample datasets in data/samples/ (SM-87, REQ-UP-9).

Four domain datasets of 200 rows, each with one deliberate defect, plus a three-class
dataset for multi-class tests. Output is deterministic, so re-running changes nothing.

    python tools/make_samples.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
ROWS = 200


def labels_from(score, rng, noise=0.6):
    """Binary labels from a linear score plus noise, so models have something to learn."""
    noisy = score + rng.normal(0, noise, len(score))
    return (noisy > np.median(noisy)).astype(int)


def banking(rng):
    """Bank churn. Defect: missing values in Balance and CreditScore."""
    age = rng.integers(18, 80, ROWS)
    balance = rng.normal(60000, 30000, ROWS).round(2).clip(0)
    credit = rng.integers(350, 850, ROWS).astype(float)
    products = rng.integers(1, 5, ROWS)
    active = rng.integers(0, 2, ROWS)
    geography = rng.choice(["France", "Germany", "Spain"], ROWS)
    score = 0.04 * (age - 45) + (geography == "Germany") * 0.8 - active * 0.9 + (products > 2)
    frame = pd.DataFrame(
        {
            "CustomerId": [f"C{100000 + i}" for i in range(ROWS)],
            "Age": age,
            "Geography": geography,
            "Gender": rng.choice(["Male", "Female"], ROWS),
            "Balance": balance,
            "CreditScore": credit,
            "NumOfProducts": products,
            "IsActiveMember": active,
            "Exited": labels_from(score, rng),
        }
    )
    frame.loc[rng.choice(ROWS, 30, replace=False), "Balance"] = np.nan
    frame.loc[rng.choice(ROWS, 15, replace=False), "CreditScore"] = np.nan
    return frame


def healthcare(rng):
    """Diabetes screening. Defect: string labels (Positive / Negative)."""
    glucose = rng.normal(120, 30, ROWS).round(1)
    bmi = rng.normal(28, 6, ROWS).round(1)
    age = rng.integers(21, 81, ROWS)
    pressure = rng.normal(72, 12, ROWS).round()
    smoker = rng.choice(["Yes", "No"], ROWS, p=[0.3, 0.7])
    score = 0.04 * (glucose - 120) + 0.08 * (bmi - 28) + 0.02 * (age - 50) + (smoker == "Yes")
    return pd.DataFrame(
        {
            "PatientId": [f"P-{i:04d}" for i in range(ROWS)],
            "Age": age,
            "Glucose": glucose,
            "BloodPressure": pressure,
            "BMI": bmi,
            "Smoker": smoker,
            "Diagnosis": np.where(labels_from(score, rng) == 1, "Positive", "Negative"),
        }
    )


def telecom(rng):
    """Telecom churn. Defect: high-cardinality City column (60 values)."""
    tenure = rng.integers(0, 72, ROWS)
    monthly = rng.normal(70, 25, ROWS).round(2).clip(18)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], ROWS, p=[0.5, 0.3, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], ROWS)
    score = -0.05 * tenure + 0.02 * (monthly - 70) + (contract == "Month-to-month") * 1.2
    return pd.DataFrame(
        {
            "Tenure": tenure,
            "MonthlyCharges": monthly,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": rng.choice(["Card", "Bank transfer", "Cheque", "UPI"], ROWS),
            "City": rng.choice([f"City_{i:02d}" for i in range(60)], ROWS),
            "Churn": np.where(labels_from(score, rng) == 1, "Yes", "No"),
        }
    )


def insurance(rng):
    """Insurance claim fraud. Defect: constant Country column."""
    premium = rng.normal(15000, 5000, ROWS).round(2).clip(2000)
    claims = rng.poisson(1.2, ROWS)
    vehicle_age = rng.integers(0, 15, ROWS)
    policy = rng.choice(["Basic", "Standard", "Premium"], ROWS)
    score = 0.6 * claims + 0.08 * vehicle_age - 0.00005 * (premium - 15000)
    return pd.DataFrame(
        {
            "Age": rng.integers(18, 75, ROWS),
            "VehicleAge": vehicle_age,
            "AnnualPremium": premium,
            "ClaimsLastYear": claims,
            "PolicyType": policy,
            "Country": "India",
            "Fraud": labels_from(score, rng),
        }
    )


def three_class(rng):
    """Customer segments with three classes, for multi-class tests."""
    spend = rng.normal(500, 200, ROWS).round(2)
    visits = rng.integers(1, 30, ROWS)
    score = 0.004 * spend + 0.1 * visits + rng.normal(0, 0.5, ROWS)
    segment = pd.qcut(score, 3, labels=["Bronze", "Silver", "Gold"]).astype(str)
    return pd.DataFrame(
        {
            "AnnualSpend": spend,
            "VisitsPerMonth": visits,
            "Channel": rng.choice(["Online", "Store", "Phone"], ROWS),
            "Segment": segment,
        }
    )


SAMPLES = {
    "banking": banking,
    "healthcare": healthcare,
    "telecom": telecom,
    "insurance": insurance,
    "three_class": three_class,
}


def main():
    """Write every sample CSV to data/samples/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for offset, (name, build) in enumerate(SAMPLES.items()):
        frame = build(np.random.default_rng(42 + offset))
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(frame)} rows, {frame.shape[1]} columns)")


if __name__ == "__main__":
    main()
