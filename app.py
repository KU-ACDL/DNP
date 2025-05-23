# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def num_motor(n_motors):
    if n_motors =="쿼드콥터(4)":
        n_motors_int = 4;
    elif n_motors =="헥사콥터(6)":
        n_motors_int = 6;
    # elif n_motors =="옥타콥터(8)":
    #     n_motors_int = 8;
    
    return n_motors_int

def prop_material(prop_M):
    
    if prop_M == "플라스틱":
        plastic = 1;
        carbon = 0;
    else:
        plastic = 0;
        carbon = 1;
    
    return plastic, carbon
    
hover = joblib.load("hover_v2_gp_model.joblib")
hover_model = hover['model']
hover_scaler = hover['scaler']
hover_feature = hover['feature_names']

flyove = joblib.load("flyover_gp_model.joblib")
flyover_model = flyove['model']
flyover_scaler = flyove['scaler']
flyover_feature = flyove['feature_names']


st.set_page_config(page_title="Konkuk Univ. ACDL - 드론 소음 예측", layout="centered")
st.title("드론 소음 예측 모델")
st.write("드론소음예측모델 ver. 0.0 (2025.05)")
# ─────────────────────────
# 상단 전체 (입력1) : 드론 스펙
# ─────────────────────────
st.subheader("[예측 입력]")
st.subheader("입력 1: 드론 스펙")
col1, col2 = st.columns(2)

with col1:
    n_motors = st.radio(
    "멀티콥터의 모터 수 : ",
    ["쿼드콥터(4)", "헥사콥터(6)"]
    )
    
    weight = st.number_input("드론 중량(kg)", min_value=0.10, max_value=25.00, value=2.0, step=0.01, format="%.2f")

with col2:
    prop_D = st.number_input("프로펠러 직경 (inch)", min_value=3.00, max_value=35.00, value=18.00, step=0.01, format="%.2f")
    prop_P = st.number_input("프로펠러 피치 (inch)", min_value=2.00, max_value=10.00, value=4.5, step=0.1, format="%.1f")
    prop_M = st.radio(
    "프로펠러 소재: ",
    ["플라스틱", "카본"]
    )

# 입력1 변수
n_motors_int = num_motor(n_motors)
plastic, carbon = prop_material(prop_M)

st.markdown("---")

# ─────────────────────────
# 하1단 전체 (입력2) : 비행임무 및 조건
# ─────────────────────────
col3, col4 = st.columns(2)
with col3:
    st.subheader("입력 2: 비행 임무")
    mission = st.radio(
        "",
        ["정지비행(호버링)", "직진비행"])

with col4:
    st.subheader("입력 3: 비행조건 입력")
    if mission == "정지비행(호버링)":
         alt = st.number_input("비행 고도(m)", min_value=1.00, max_value = 15.00, value = 5.00, step=0.01, format="%.2f")
         dist = st.number_input("이격 거리(m)", min_value=2.00, max_value = 10.00, value = 5.00, step=0.01, format="%.2f")
         
         input_dict = {
             'TEST_env_field': 1,
             'W' : weight,
             'MOT_NUM': n_motors_int,
             'PROP_D': prop_D,
             'PROP_P': prop_P,
             'ALT' : alt,
             'DIST' : dist,
             'PROP_M_Plastic' : plastic,
             'PROP_M_carbon' : carbon
             }
         
         model = hover_model
         scaler = hover_scaler
         feature = hover_feature
    else:
        alt = st.number_input("비행 고도(m)", min_value=1.00, max_value = 15.00, value = 5.00, step=0.01, format="%.2f")
        dist = st.number_input("이격 거리(m)", min_value=2.00, max_value = 10.00, value = 5.00, step=0.01, format="%.2f")
        spd = st.number_input("비행 속도(m/s)", min_value=2.00, max_value = 15.00, value = 5.00, step=0.01, format="%.2f")
        
        input_dict = {
            'W' : weight,
            'MOT_NUM': n_motors_int,
            'PROP_D': prop_D,
            'PROP_P': prop_P,
            'ALT' : alt,
            'DIST' : dist,
            'SPD' : spd,
            'PROP_M_Plastic' : plastic,
            'PROP_M_carbon' : carbon
            }
        
        model = flyover_model
        scaler = flyover_scaler
        feature = flyover_feature


for col in feature:
    if col not in input_dict:
        input_dict[col] = 0
        

input_point = pd.DataFrame([input_dict])[feature]
input_scaled = scaler.transform(input_point)
prediction, sig = model.predict(input_scaled, return_std=True)
res = float(prediction)

st.markdown("---")
st.subheader("[예측 결과]")

if mission == "정지비행(호버링)":
    st.markdown(
    f"<h3> 예측 등가소음도 : {res:.2f} dBA</h3>",
    unsafe_allow_html=True)
    
    vary_key = ['DIST', 'ALT']
        
    for key in vary_key:
        print(key)
        if key == 'DIST':
            vary_range = np.linspace(2, 10, 50)
            key0 = "Distance"
        elif key == 'ALT':
            vary_range = np.linspace(2, 15, 50)
            key0 = "Altitude"
        
        preds = []
        for v in vary_range:
            temp_input = input_dict.copy()
            temp_input[key] = v
            input_df = pd.DataFrame([temp_input])[feature]  # feature: 원래 학습한 feature 순서
            input_scaled = scaler.transform(input_df)
            y_pred = model.predict(input_scaled)
            preds.append(y_pred[0])
        
        fig, ax = plt.subplots()
        ax.scatter(input_point[key], res, s=50, c="red", label="Predicted Result")
        ax.plot(vary_range, preds, lw=2, color="darkblue", label="Prediction Mean")
        # ax.fill_between(vary_range.ravel(), 
        #                 preds - 1.96 * sig, 
        #                 preds + 1.96 *sig, 
        #                 color='lightblue', 
        #                 alpha=0.5, 
        #                 label="95% Confidence Interval")
        ax.fill_between(vary_range.ravel(), 
                        preds - 2.576 * sig, 
                        preds + 2.576 *sig, 
                        color='orange', 
                        alpha=0.5, 
                        label="99% Confidence Interval")
        ax.set_ylim(40,100)
        ax.set_xlabel(f"{key0}")
        ax.set_ylabel("Noise Level (dB)")
        ax.set_title(f"Noise Prediction Results by {key0} ")
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        

else:
    st.markdown(
    f"<h3> 예측 최고소음도 : {res:.2f} dBA</h3>",
    unsafe_allow_html=True)
    
    vary_key = ['DIST', 'ALT', 'SPD']
        
    for key in vary_key:
        print(key)
        if key == 'DIST':
            vary_range = np.linspace(2, 10, 50)
            key0 = "Distance"
        elif key == 'ALT':
            vary_range = np.linspace(2, 15, 50)
            key0 = "Altitude"
        elif key == 'SPD':
            vary_range = np.linspace(4, 10, 50)
            key0 = "Speed"
            
        preds = []
        for v in vary_range:
            temp_input = input_dict.copy()
            temp_input[key] = v
            input_df = pd.DataFrame([temp_input])[feature]  # feature: 원래 학습한 feature 순서
            input_scaled = scaler.transform(input_df)
            y_pred = model.predict(input_scaled)
            preds.append(y_pred[0])
        
        fig, ax = plt.subplots()
        ax.scatter(input_point[key], res, s=50, c="red", label="Predicted Result")
        ax.plot(vary_range, preds, lw=2, color="darkblue", label="Prediction Mean")
        # ax.fill_between(vary_range.ravel(), 
        #                 preds - 1.96 * sig, 
        #                 preds + 1.96 *sig, 
        #                 color='lightblue', 
        #                 alpha=0.5, 
        #                 label="95% Confidence Interval")
        ax.fill_between(vary_range.ravel(), 
                        preds - 2.576 * sig, 
                        preds + 2.576 *sig, 
                        color='orange', 
                        alpha=0.5, 
                        label="99% Confidence Interval")
        ax.set_ylim(40,100)
        ax.set_xlabel(f"{key0}")
        ax.set_ylabel("Noise Level (dB)")
        ax.set_title(f"Noise Prediction Results by {key0} ")
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)



st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>© 2025 KUACDL | Aerospace Computing & Design Lab.</div>",
    unsafe_allow_html=True
)

