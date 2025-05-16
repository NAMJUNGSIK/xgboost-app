import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime

# --- 데이터 불러오기 ---
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Tomorrow'] = df['Close'].shift(-1)
    df = df.dropna(subset=['Tomorrow']) 
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    df = df.dropna()
    return df

# --- XGBoost 모델 학습 ---
def train_xgb_model(df):
    features = df[['Close', 'Return', 'MA5', 'MA10']]
    labels = df['Target']
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(features, labels)
    return model

# --- 예측 함수 ---
def predict_next_day(ticker, start_date, end_date):
    df = load_data(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("데이터가 충분하지 않습니다.")
        return None

    model = train_xgb_model(df)

    latest_features = df[['Close', 'Return', 'MA5', 'MA10']].iloc[-1].values.reshape(1, -1)
    prob = model.predict_proba(latest_features)[0][1]  # 상승 확률

    return round(prob * 100, 2)

# --- Streamlit UI ---
st.title("📊 XGBoost 기반 주가 상승 확률 예측")

ticker = st.text_input("종목 코드 입력 (예: 005930.KS - 삼성전자)", "005930.KS")
start_date = st.date_input("시작 날짜", datetime(2020, 1, 1))
end_date = st.date_input("종료 날짜", datetime.today())

if st.button("예측 시작"):
    with st.spinner("데이터 수집 및 모델 학습 중..."):
        prob = predict_next_day(ticker, start_date, end_date)
        if prob is not None:
            st.success(f"📈 내일 종가 상승 확률: **{prob}%**")
