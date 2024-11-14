import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import random

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Historic Waiting List", "Referrals Prediction", "Capacity"])

    if page == "Home":
        home_page()
    elif page == "Historic Waiting List":
        historic_waiting_list_page()
    elif page == "Referrals Prediction":
        referrals_prediction_page()
    elif page == "Capacity":
        capacity_page()

def home_page():
    st.title("Upload and Specialty Selection")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("Dataset loaded successfully!")
        if "specialty" in df.columns:
            specialty = st.selectbox("Choose Specialty", df["specialty"].unique())
            st.write(f"Selected Specialty: {specialty}")
        else:
            st.error("No 'specialty' column found in dataset.")

def historic_waiting_list_page():
    st.title("Historic Waiting List")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)
        if "date" in df.columns and "referrals" in df.columns and "removals" in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            monthly_data = df.resample('M').sum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['referrals'], mode='lines', name='Referrals'))
            fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['removals'], mode='lines', name='Removals'))
            fig.update_layout(title="Monthly Referrals vs. Removals", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig)

            months = st.number_input("Enter number of months to use as baseline", min_value=1, value=6)
            forecast_months = st.number_input("Months to predict forward", min_value=1, value=12)

            baseline_data = monthly_data['referrals'][-months:]
            predictions = [random.choice(baseline_data) for _ in range(forecast_months)]
            forecast_df = pd.DataFrame({
                "Predicted Referrals": predictions
            }, index=pd.date_range(monthly_data.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='M'))

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted Referrals'], mode='lines+markers', name='Predicted Referrals'))
            fig_forecast.update_layout(title="Predicted Referrals", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig_forecast)

def referrals_prediction_page():
    st.title("Referrals Prediction")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)
        if "date" in df.columns and "referrals" in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            monthly_data = df['referrals'].resample('M').sum()

            X = np.arange(len(monthly_data)).reshape(-1, 1)
            y = monthly_data.values
            model = LinearRegression().fit(X, y)
            predictions_regression = model.predict(X)
            avg_forecast = [monthly_data.mean()] * len(monthly_data)

            mse_regression = mean_squared_error(y, predictions_regression)
            mse_avg = mean_squared_error(y, avg_forecast)
            best_model = "Regression" if mse_regression < mse_avg else "Average"

            st.write(f"Best model for recent data: {best_model}")

            future_X = np.arange(len(monthly_data), len(monthly_data) + 12).reshape(-1, 1)
            forecast = model.predict(future_X) if best_model == "Regression" else [monthly_data.mean()] * 12

            forecast_df = pd.DataFrame({
                "Predicted Referrals": forecast
            }, index=pd.date_range(monthly_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M'))

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted Referrals'], mode='lines+markers', name='Predicted Referrals'))
            fig_forecast.update_layout(title="12-Month Referrals Forecast", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig_forecast)

def capacity_page():
    st.title("Capacity Analysis")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)
        if "date" in df.columns and "first_appointments" in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            monthly_first_appointments = df['first_appointments'].resample('M').sum()

            referral_forecast = [monthly_first_appointments.mean()] * 12
            referral_df = pd.DataFrame({
                "Predicted Referrals": referral_forecast
            }, index=pd.date_range(monthly_first_appointments.index[-1] + pd.DateOffset(months=1), periods=12, freq='M'))

            ratio = st.number_input("First-to-follow-up ratio", min_value=0.1, value=2.0)
            follow_up_forecast = [x * ratio for x in referral_forecast]
            follow_up_df = pd.DataFrame({
                "Required Follow-ups": follow_up_forecast
            }, index=pd.date_range(monthly_first_appointments.index[-1] + pd.DateOffset(months=1), periods=12, freq='M'))

            fig_referrals = go.Figure()
            fig_referrals.add_trace(go.Scatter(x=referral_df.index, y=referral_df['Predicted Referrals'], mode='lines+markers', name='Predicted Referrals'))
            fig_referrals.update_layout(title="Predicted Referrals for Next 12 Months", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig_referrals)

            fig_follow_ups = go.Figure()
            fig_follow_ups.add_trace(go.Scatter(x=follow_up_df.index, y=follow_up_df['Required Follow-ups'], mode='lines+markers', name='Required Follow-ups'))
            fig_follow_ups.update_layout(title="Required Follow-up Appointments", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig_follow_ups)

if __name__ == "__main__":
    main()
