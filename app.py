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
