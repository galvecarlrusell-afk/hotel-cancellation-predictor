import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils import (MONTHS, explain_booking, invalid_rows, predicted_label,
                   revenue_at_risk, risk_level, validate_booking)

HERE = Path(__file__).parent
RISK_COLORS = {"HIGH": "#c62828", "MEDIUM": "#ef6c00", "LOW": "#2e7d32"}
ROW_COLORS = {"HIGH": "background-color: #f8d7da", "MEDIUM": "background-color: #fff3cd",
              "LOW": "background-color: #d4edda"}

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Booking Cancellation Predictor")

# ---------- load saved models once ----------
if not (HERE / "models" / "info.json").exists():
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load_everything():
    info = json.loads((HERE / "models" / "info.json").read_text())
    models = {m["label"]: joblib.load(HERE / m["file"]) for m in info["models"]}
    return info, models


@st.cache_data
def read_results(name):
    return pd.read_csv(HERE / "results" / name)


info, models = load_everything()
names = list(models)

# ---------- sidebar ----------
choice = st.sidebar.selectbox("Model", names, index=names.index(info["best_model"]))
model = models[choice]
st.sidebar.caption(f"Best model in cross-validation: {info['best_model']}")
currency = st.sidebar.selectbox("Currency symbol", ["€", "₱", "$", ""], index=0,
                                help="Only changes how money is displayed. The dataset does not "
                                     "state its currency.")
st.sidebar.markdown("**Risk levels**")
medium_cut = st.sidebar.slider("MEDIUM risk from", 0.05, 0.85, 0.35, 0.05, format="%.2f")
high_cut = st.sidebar.slider("HIGH risk from", float(medium_cut) + 0.05, 0.95,
                             min(max(0.60, float(medium_cut) + 0.05), 0.95), 0.05, format="%.2f")
st.sidebar.write("Enter a booking, click Predict, and read the cancellation probability. "
                 "Results are automated estimates, not guarantees.")


def money(x):
    return f"{currency}{x:,.0f}"


def predict_p(df, m=None):
    """Cancellation probability for each row of df."""
    return (m or model).predict_proba(df[info["columns"]])[:, 1]


def level_of(p):
    return risk_level(p, medium_cut, high_cut)


def badge(level):
    return (f"<span style='background:{RISK_COLORS[level]};color:white;padding:6px 18px;"
            f"border-radius:999px;font-weight:700;font-size:1.1rem'>{level} RISK</span>")


tab1, tab2, tab3, tab4 = st.tabs(["Single booking", "Batch (CSV)", "Data insights", "Model results"])

# ---------- single booking ----------
DEFAULTS = {"adults": 2, "children": 0, "weekend": 1, "week": 2, "lead": 60, "month": "June",
            "price": 100.0, "requests": 0, "prev_c": 0, "prev_nc": 0,
            "meal": info["categories"]["type_of_meal_plan"][0],
            "room": info["categories"]["room_type_reserved"][0],
            "segment": info["categories"]["market_segment_type"][-1],
            "parking": False, "repeated": False}
PRESETS = {   # checked against the trained models: "low" is about 0-2% risk, "high" about 70-85%
    "low": {**DEFAULTS, "lead": 30, "requests": 2, "segment": "Corporate", "price": 80.0,
            "repeated": True, "prev_nc": 2, "parking": True},
    "high": {**DEFAULTS, "lead": 130, "requests": 0, "price": 115.0, "month": "September",
             "week": 3, "weekend": 1},
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_preset(name):
    for k, v in (PRESETS[name] if name != "reset" else DEFAULTS).items():
        st.session_state[k] = v


with tab1:
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    b1.button("Load low-risk example", on_click=load_preset, args=("low",))
    b2.button("Load high-risk example", on_click=load_preset, args=("high",))
    b3.button("Reset form", on_click=load_preset, args=("reset",))

    with st.form("booking_form"):          # the page only reruns when Predict is clicked
        c1, c2, c3 = st.columns(3)
        adults = c1.number_input("Adults", 0, 10, key="adults", help="At least 1 adult is required.")
        children = c1.number_input("Children", 0, 10, key="children")
        weekend = c1.number_input("Weekend nights", 0, 20, key="weekend", help="Friday and Saturday nights.")
        week = c1.number_input("Week nights", 0, 40, key="week", help="Sunday to Thursday nights.")
        lead = c1.number_input("Lead time (days)", 0, 800, key="lead",
                               help="Days between booking and arrival. Longer lead time usually means higher risk.")
        month_name = c2.selectbox("Arrival month", MONTHS, key="month")
        price = c2.number_input("Average price per room (per night)", 0.0, 1000.0, key="price")
        requests = c2.number_input("Special requests", 0, 10, key="requests",
                                   help="Guests who make special requests are usually more committed.")
        prev_c = c2.number_input("Previous cancellations", 0, 50, key="prev_c")
        prev_nc = c2.number_input("Previous bookings not cancelled", 0, 100, key="prev_nc")
        meal = c3.selectbox("Meal plan", info["categories"]["type_of_meal_plan"], key="meal")
        room = c3.selectbox("Room type", info["categories"]["room_type_reserved"], key="room")
        segment = c3.selectbox("Market segment", info["categories"]["market_segment_type"], key="segment")
        parking = int(c3.checkbox("Requires parking", key="parking"))
        repeated = int(c3.checkbox("Repeated guest", key="repeated"))
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        problems = validate_booking(adults, weekend, week, price)
        if problems:
            for msg in problems:
                st.error(msg)
        else:
            month = MONTHS.index(month_name) + 1
            row = pd.DataFrame([{
                "no_of_adults": adults, "no_of_children": children,
                "no_of_weekend_nights": weekend, "no_of_week_nights": week,
                "type_of_meal_plan": meal, "required_car_parking_space": parking,
                "room_type_reserved": room, "lead_time": lead, "arrival_month": month,
                "market_segment_type": segment, "repeated_guest": repeated,
                "no_of_previous_cancellations": prev_c,
                "no_of_previous_bookings_not_canceled": prev_nc,
                "avg_price_per_room": price, "no_of_special_requests": requests}])
            p = float(predict_p(row)[0])
            level, action = level_of(p)

            # revenue at risk = booking value x probability of cancellation
            nights = weekend + week
            value = price * nights
            m1, m2, m3 = st.columns(3)
            m1.metric("Cancellation probability", f"{p:.1%}")
            m2.metric("Booking value", money(value))
            m3.metric("Revenue at risk", money(revenue_at_risk(price, nights, p)))
            st.subheader(f"Prediction: {predicted_label(p)}")
            st.markdown(badge(level), unsafe_allow_html=True)
            st.progress(p)
            st.write(f"**{action}**")

            # why: which inputs push this booking above/below a typical one
            left, right = st.columns(2)
            with left:
                st.write("**Why this estimate?**")
                why = explain_booking(predict_p, row[info["columns"]], info["reference"])
                if why.empty:
                    st.caption("This booking looks like a typical one.")
                else:
                    why["effect (points)"] = (why["effect"] * 100).round(1)
                    st.bar_chart(why.set_index("feature")["effect (points)"], horizontal=True)
                    st.caption("Positive = this input raises the risk compared with a typical booking "
                               "(median / most common value). Negative = lowers it.")
            with right:
                st.write("**All models for this booking**")
                st.table(pd.DataFrame([{"Model": n, "Probability": f"{float(predict_p(row, m)[0]):.1%}",
                                        "Risk": level_of(float(predict_p(row, m)[0]))[0]}
                                       for n, m in models.items()]))

            # what-if: change one thing and see how the estimate moves
            scenarios = {"Add 1 special request": {"no_of_special_requests": min(requests + 1, 10)},
                         "Room price 10% lower": {"avg_price_per_room": round(price * 0.9, 2)}}
            if lead > 0:
                scenarios["Arrival 30 days sooner (lead time -30)"] = {"lead_time": max(lead - 30, 0)}
            rows = []
            for label, change in scenarios.items():
                alt = row.copy()
                for col, val in change.items():
                    alt[col] = val
                p2 = float(predict_p(alt)[0])
                rows.append({"What if...": label, "New probability": f"{p2:.1%}",
                             "Change": f"{(p2 - p) * 100:+.1f} points"})
            st.write("**What-if analysis**")
            st.table(pd.DataFrame(rows))
            st.caption(f"Predicted by {choice}. The explanation and what-if tables show how the model's "
                       "estimate changes. They are not proof that a change would prevent a cancellation.")


# ---------- batch CSV ----------
def run_batch(upload):
    """Validate an uploaded CSV and show predictions. Returns early on errors
    (st.stop() would also hide the other tabs)."""
    try:
        df = pd.read_csv(upload)
    except Exception:
        st.error("Could not read that file. Please upload a valid CSV.")
        return
    missing = [c for c in info["columns"] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    df = df[info["columns"]].copy()
    for c in info["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = invalid_rows(df)
    if bad.any():
        st.warning(f"{int(bad.sum())} invalid row(s) skipped.")
    ok = df[~bad].copy()
    if ok.empty:
        st.error("No valid rows to predict.")
        return
    ok["cancel_probability"] = predict_p(ok).round(3)
    ok["prediction"] = ok["cancel_probability"].apply(predicted_label)
    ok["risk"] = ok["cancel_probability"].apply(lambda p: level_of(p)[0])
    ok["booking_value"] = (ok["avg_price_per_room"] * (
        ok["no_of_weekend_nights"] + ok["no_of_week_nights"])).round(2)
    ok["revenue_at_risk"] = (ok["booking_value"] * ok["cancel_probability"]).round(2)
    ok = ok.sort_values("cancel_probability", ascending=False)

    s1, s2, s3 = st.columns(3)
    s1.metric("Bookings checked", len(ok))
    s2.metric("High-risk bookings", int((ok["risk"] == "HIGH").sum()))
    s3.metric("Total revenue at risk", money(ok["revenue_at_risk"].sum()))

    counts = ok["risk"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0).astype(int)
    st.write("**Risk distribution**")
    st.bar_chart(counts)
    show = st.multiselect("Show risk levels", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    view = ok[ok["risk"].isin(show)]
    st.dataframe(view.style.map(lambda v: ROW_COLORS.get(v, ""), subset=["risk"]),
                 use_container_width=True)
    st.download_button("Download shown predictions", view.to_csv(index=False),
                       "predictions.csv", "text/csv")


with tab2:
    st.write("Upload a CSV with these columns:")
    st.code(", ".join(info["columns"]))
    template = pd.DataFrame([{c: (info["categories"][c][0] if c in info["categories"] else 1)
                              for c in info["columns"]}])
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "booking_template.csv", "text/csv")
    up = st.file_uploader("Upload bookings CSV", type="csv")
    if up is not None:
        run_batch(up)

# ---------- data insights ----------
with tab3:
    st.write("Cancellation rate in the training data, by group. Bars show the share of bookings "
             "that were cancelled; the overall rate is "
             f"{info['canceled_share']:.1%}.")
    ins = {"Lead time": "insights_lead_time.csv", "Market segment": "insights_segment.csv",
           "Special requests": "insights_requests.csv", "Arrival month": "insights_month.csv"}
    cols = st.columns(2)
    for i, (title, file) in enumerate(ins.items()):
        data = read_results(file)
        with cols[i % 2]:
            st.write(f"**{title}**")
            st.bar_chart(data.set_index("group")["cancel_rate"])
            st.caption("Bookings per group: " + ", ".join(
                f"{g} ({n:,})" for g, n in zip(data["group"], data["bookings"])))
    st.caption("These are patterns in one hotel group's history, not proof of cause and effect.")

# ---------- model results ----------
with tab4:
    st.write(f"Trained on {info['rows_used']:,} cleaned bookings "
             f"({info['canceled_share']:.1%} cancelled). Test set = 20% of the data. "
             "Metrics treat Canceled as the positive class.")
    comp = read_results("model_comparison.csv")
    st.write("**Test-set results**")
    st.dataframe(comp, use_container_width=True)
    st.write("**5-fold cross-validation on the training set** (mean +/- standard deviation)")
    st.dataframe(read_results("cv_results.csv"), use_container_width=True)

    mrow = comp[comp["Model"] == choice].iloc[0]
    with st.expander("What do these numbers mean? (plain language)"):
        st.write(f"- **Recall {mrow['Recall']:.0%}** - of every 100 bookings that really get cancelled, "
                 f"{choice} catches about {mrow['Recall'] * 100:.0f}.")
        st.write(f"- **Precision {mrow['Precision']:.0%}** - of every 100 bookings it flags as "
                 f"'will cancel', about {mrow['Precision'] * 100:.0f} really are cancelled; the rest are false alarms.")
        st.write(f"- **F1 {mrow['F1']:.2f}** - one score that balances recall and precision.")
        st.write(f"- **Accuracy {mrow['Accuracy']:.0%}** - share of all bookings classified correctly. "
                 "On its own it can mislead because most bookings are not cancelled.")
        st.write(f"- **ROC-AUC {mrow['ROC-AUC']:.2f}** - how well the model ranks cancelled bookings above "
                 "kept ones (0.5 = guessing, 1.0 = perfect).")

    slug = next(m["slug"] for m in info["models"] if m["label"] == choice)
    left, right = st.columns(2)
    left.image(str(HERE / "results" / f"confusion_{slug}.png"), caption=f"Confusion matrix: {choice}")
    with right:
        st.write("Top factors (Random Forest, built-in)")
        st.bar_chart(read_results("feature_importance.csv").set_index("feature")["importance"])
    c1, c2 = st.columns(2)
    c1.image(str(HERE / "results" / "roc_curves.png"), caption="ROC curves (AUC in legend)")
    c2.image(str(HERE / "results" / "pr_curves.png"), caption="Precision-recall curves")