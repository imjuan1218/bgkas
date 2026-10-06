import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 레이아웃 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울의 과거 기온 관측 데이터를 바탕으로 연도별 평균기온 추세를 분석하고 기온을 예측합니다.")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_preprocess_data():
    # 데이터 불러오기 (UTF-8 인코딩)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 날짜 컬럼 파싱 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 대상
    df_filtered = df[df["연도"] <= 2025]
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_df = df_filtered.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일수가 300일 미만인 해 제외
    yearly_clean = yearly_df[yearly_df["관측일수"] >= 300].copy()
    
    return yearly_clean

df_yearly = load_and_preprocess_data()

# 2. 독립 변수 설정 및 회귀선 계산
# 1908년부터 지난 연수를 독립 변수로 설정 (1908년 = 0)
df_yearly["X"] = df_yearly["연도"] - 1908

# 최소자승법을 통한 1차 회귀선 계산 (기울기, 절편)
slope, intercept = np.polyfit(df_yearly["X"], df_yearly["평균기온"], 1)

# 연도와 평균기온 간의 상관계수 계산
corr = np.corrcoef(df_yearly["연도"], df_yearly["평균기온"])[0, 1]

# 데이터 주요 통계값 추출
num_years = len(df_yearly)
start_year = int(df_yearly["연도"].min())
end_year = int(df_yearly["연도"].max())

# 3. 화면 상단 요약 정보 표시
st.markdown("### 📊 학습 데이터 요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("회귀선 생성 연도 수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{corr:.4f}")

st.markdown("---")

# 4. 연도 선택 슬라이더 및 예상 기온 계산 (1900 ~ 2100)
st.markdown("### 🔮 기온 예측하기")
selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

# 선택한 연도의 독립변수 값 및 예상 기온 계산
selected_x = selected_year - 1908
predicted_temp = slope * selected_x + intercept

# 예상 기온 크게 표시
st.metric(
    label=f"📌 {selected_year}년 예상 연평균 기온",
    value=f"{predicted_temp:.2f} °C"
)

# 5. Plotly 시각화 (산점도 + 회귀선 + 선택 연도 포인트)
fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=df_yearly["연도"],
        y=df_yearly["평균기온"],
        mode="markers",
        name="실제 관측 연평균기온",
        marker=dict(color="#1f77b4", size=8, opacity=0.8),
        hovertemplate="%{x}년 관측 평균기온: %{y:.2f}°C<extra></extra>"
    )
)

# 전체 슬라이더 범위(1900~2100년)에 대한 회귀 직선
all_years = np.arange(1900, 2101)
all_preds = slope * (all_years - 1908) + intercept

fig.add_trace(
    go.Scatter(
        x=all_years,
        y=all_preds,
        mode="lines",
        name="회귀 직선",
        line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="%{x}년 추세선 기온: %{y:.2f}°C<extra></extra>"
    )
)

# 선택된 연도의 예측값 강조 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers+text",
        name=f"선택 연도 ({selected_year}년)",
        marker=dict(color="#ff7f0e", size=14, symbol="star"),
        text=[f"{predicted_temp:.2f}°C"],
        textposition="top center",
        hovertemplate="선택 연도: %{x}년<br>예상 기온: %{y:.2f}°C<extra></extra>"
    )
)

# 그래프 레이아웃 설정
fig.update_layout(
    title="서울 연도별 평균기온 분포 및 회귀선 추세",
    xaxis=dict(title="연도", dtick=20),
    yaxis=dict(title="평균기온 (°C)"),
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 로드 및 정제
url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
df = pd.read_csv(url, encoding="utf-8")
df.columns = df.columns.str.strip()

df["날짜"] = pd.to_datetime(df["날짜"])
df["연도"] = df["날짜"].dt.year

# 2025년 이하, 관측일수 300일 이상 필터링
df_filtered = df[df["연도"] <= 2025]
yearly = df_filtered.groupby("연도").agg(
    관측일수=("평균기온", "count"),
    평균기온=("평균기온", "mean")
).reset_index()

df_clean = yearly[yearly["관측일수"] >= 300].copy()

# 2. 데이터셋 분할
# 공통 테스트 데이터: 최근 20년 (2006 ~ 2025)
test_df = df_clean[(df_clean["연도"] >= 2006) & (df_clean["연도"] <= 2025)]

# 훈련 데이터 A: 최근 50년 (1956 ~ 2005)
train_50_df = df_clean[(df_clean["연도"] >= 1956) & (df_clean["연도"] <= 2005)]

# 훈련 데이터 B: 최근 100년 (1906 ~ 2005)
train_100_df = df_clean[(df_clean["연도"] >= 1906) & (df_clean["연도"] <= 2005)]

# 3. 모델 학습 및 평가 함수
def evaluate_linear_model(train_data, eval_data, model_name="Model"): 
    X_train = train_data[["연도"]]
    y_train = train_data["평균기온"]
    
    X_eval = eval_data[["연도"]]
    y_eval = eval_data["평균기온"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_eval)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    mae = mean_absolute_error(y_eval, y_pred)
    mse = mean_squared_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)
    
    return {
        "모델": model_name,
        "기울기(Slope)": slope,
        "절편(Intercept)": intercept,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }

# 4. 모델 평가 실행
# 전체 데이터 모델 (전체 데이터 학습 & 전체 데이터 평가)
res_all = evaluate_linear_model(df_clean, df_clean, "전체 데이터 (1908~2025)")

# 최근 50년 학습 모델 (1956~2005 학습 -> 2006~2025 테스트)
res_50 = evaluate_linear_model(train_50_df, test_df, "최근 50년 (1956~2005)")

# 최근 100년 학습 모델 (1906~2005 학습 -> 2006~2025 테스트)
res_100 = evaluate_linear_model(train_100_df, test_df, "최근 100년 (1906~2005)")

# 결과 출력
results_df = pd.DataFrame([res_all, res_50, res_100])
print(results_df.to_string(index=False)) 


