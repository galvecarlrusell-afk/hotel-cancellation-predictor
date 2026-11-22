"""Pure helper functions (no Streamlit) so they can be unit-tested."""
import numpy as np
import pandas as pd

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

HIGH_RISK = 0.60
MEDIUM_RISK = 0.35


def risk_level(p, medium=MEDIUM_RISK, high=HIGH_RISK):
    """Return (label, suggested action) for a cancellation probability."""
    if p >= high:
        return "HIGH", "Send a reminder or ask for a deposit."
    if p >= medium:
        return "MEDIUM", "Send a confirmation reminder."
    return "LOW", "No special action needed."


def predicted_label(p, threshold=0.5):
    """Class label shown to the user: 'Likely to Cancel' or 'Likely to Stay'."""
    return "Likely to Cancel" if p >= threshold else "Likely to Stay"


def validate_booking(adults, weekend_nights, week_nights, price):
    """Return a list of problems for one booking (empty list = valid)."""
    problems = []
    if adults < 1:
        problems.append("At least 1 adult is required.")
    if weekend_nights + week_nights < 1:
        problems.append("The stay must be at least 1 night.")
    if price <= 0:
        problems.append("Price must be greater than 0.")
    return problems


def invalid_rows(df):
    """Boolean Series, True for rows that cannot be predicted (batch upload)."""
    nights = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    return (df.isna().any(axis=1)
            | (df["no_of_adults"] < 1)
            | (nights < 1)
            | (df["avg_price_per_room"] <= 0)
            | ~df["arrival_month"].between(1, 12))


def revenue_at_risk(price, nights, probability):
    """Booking value x probability of cancellation."""
    return price * nights * probability


def explain_booking(predict_fn, row, reference, top=8):
    """How much each input moves this booking's risk compared with a typical booking.

    Each feature is set to its typical value (median / most common) one at a time.
    effect = p(booking) - p(booking with that feature typical), in probability points.
    Positive = this booking's value pushes the risk up. Works with any model.
    """
    base = float(predict_fn(row)[0])
    alts = []
    for col in row.columns:
        alt = row.copy()
        alt[col] = reference[col]
        alts.append(alt)
    probs = predict_fn(pd.concat(alts, ignore_index=True))
    out = pd.DataFrame({"feature": list(row.columns), "effect": base - np.asarray(probs)})
    out = out[out["effect"].abs() > 1e-9]
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index).head(top)