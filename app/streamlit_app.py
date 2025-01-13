import streamlit as st
import joblib
import numpy as np


def main():
    st.title("Fraud Detection System – Demo")

    st.write("""
    Enter your transaction details and the model will assess whether it is likely to be fraudulent.
    """)

    model_path = "C:/Users/kamil/Documents/pythonProject1/fraud_detection/models/best_random_forest.pkl"
    model = joblib.load(model_path)

    distance_from_home = st.number_input("distance_from_home", min_value=0.0, max_value=100000.0, value=10.0)
    distance_from_last_transaction = st.number_input("distance_from_last_transaction", min_value=0.0,
                                                     max_value=100000.0, value=0.5)
    ratio_to_median_purchase_price = st.number_input("ratio_to_median_purchase_price", min_value=0.0, max_value=1000.0,
                                                     value=1.0)

    repeat_retailer = st.selectbox("repeat_retailer (1 – yes, 0 – no)", (0, 1))
    used_chip = st.selectbox("used_chip (1 – yes, 0 – no)", (0, 1))
    used_pin_number = st.selectbox("used_pin_number (1 – yes, 0 – no)", (0, 1))
    online_order = st.selectbox("online_order (1 – yes, 0 – no)", (0, 1))

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Check the transaction"):
        features = np.array([[
            distance_from_home,
            distance_from_last_transaction,
            ratio_to_median_purchase_price,
            repeat_retailer,
            used_chip,
            used_pin_number,
            online_order
        ]])

        scaler_path = "C:/Users/kamil/Documents/pythonProject1/fraud_detection/models/scaler.pkl"
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(features)
        prob_fraud = model.predict_proba(features_scaled)[:, 1][0]

        if prob_fraud >= threshold:
            st.error(f"Fraud (p = {prob_fraud:.4f})")
        else:
            st.success(f"Normal (p = {prob_fraud:.4f})")

        st.write("Probability of fraud:", round(prob_fraud, 4))


if __name__ == "__main__":
    main()
