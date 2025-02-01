# -*- coding: utf-8 -*-
"""
데이터 시각화 및 탐색적 데이터 분석, 모델 학습 및 평가, 그리고 평가 지표 플롯 그리기
"""

import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

# Plotly 관련 (노트북 환경에서 interactive plot을 원할 때 사용)
from plotly.offline import init_notebook_mode
init_notebook_mode()

# scikit-learn 관련 라이브러리
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    explained_variance_score, max_error, mean_squared_error, 
    mean_absolute_error, r2_score, median_absolute_error
)

# 모델 저장용 joblib
import joblib

### 1. 데이터 로드 및 전처리 ###
# CSV 파일 읽기
df_usage = pd.read_csv('./dataset/train/power.csv')
df_weather = pd.read_csv('./dataset/train/weather.csv')

# --- df_usage 날짜 처리 ---
n = df_usage.shape[0]
p1 = pd.Series(range(n), pd.period_range('2016-06-01 00:00:00', freq='1H', periods=n))
df_usage['StartDate'] = p1.to_frame().index
df_usage['StartDate'] = df_usage['StartDate'].apply(lambda x: x.to_timestamp())
df_usage['Date'] = pd.DatetimeIndex(df_usage['StartDate']).date

# --- df_weather 날짜 처리 ---
m = df_weather.shape[0]
p2 = pd.Series(range(m), pd.period_range('2016-06-01', freq='1D', periods=m))
df_weather['Date'] = p2.to_frame().index
df_weather['Date'] = df_weather['Date'].apply(lambda x: x.to_timestamp())

# --- 일별 전력 사용량 집계 ---
df_usage_daily = df_usage.groupby('Date').sum(numeric_only=True)
df_usage_daily['day_of_week'] = df_usage_daily['day_of_week'].apply(lambda x: x / 24)
notes_col = df_usage.groupby('Date').first()['notes'].values
df_usage_daily['notes'] = notes_col

# --- weather 데이터와 결합 ---
k = df_usage_daily.shape[0]
df1 = df_usage_daily['Value (kWh)'].values
comb_df = df_weather.iloc[0:k].copy()
comb_df['kWh_usage'] = pd.Series(df1).to_frame()
comb_df['notes'] = notes_col

# 상관분석에 사용할 데이터프레임 (문자열 컬럼 제외 후)
corr_df = comb_df[['Temp_max', 'Temp_min', 'Dew_max', 'Dew_min', 'kWh_usage', 'notes']]

# --- 24시간 단위로 전력 사용량 분할 (예: 하루치 배열) ---
power_split = df_usage['Value (kWh)'].values
split_data_power = [power_split[x:x+24] for x in range(0, len(power_split), 24)]
comb_df['power_array_nsaled'] = pd.Series(split_data_power).to_frame()

### 2. EDA: PairGrid 및 Heatmap 저장 ###
# PairGrid: 각 변수의 분포 및 변수간 관계 (notes에 따라 색 구분)
g = sns.PairGrid(corr_df, hue="notes")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
g.savefig('./static/result/pairplot.png')
plt.close()

# Heatmap: 수치형 변수들 간 상관계수 (notes 컬럼 제외)
plt.figure(figsize=(10, 10))
sns.heatmap(corr_df.drop(columns='notes').corr(), annot=True, cmap='coolwarm', annot_kws={"color": "black", "size": 12})
plt.tight_layout()
plt.savefig('./static/result/heatmap.png')
plt.close()

### 3. 모델 학습 및 평가 ###
# 평가 지표 계산 함수
def evaluation(model_name, y_pred, y_true):
    """
    Input:
      - model_name: 문자열, 모델 이름
      - y_pred: 모델 예측값
      - y_true: 실제 레이블
    Output:
      - 평가 지표가 담긴 DataFrame (설명 분산, 최대 오차, MSE, MAE, R², 중앙 절대 오차)
    """
    data = [
        explained_variance_score(y_true, y_pred),
        max_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred, multioutput='uniform_average'),
        median_absolute_error(y_true, y_pred)
    ]
    row_index = ['Exp_Var_Score', 'Max_Error', 'MSE', 'MAE', 'R2_Score', 'Median_Abs_Error']
    return pd.DataFrame(data, columns=[model_name], index=row_index)

# 특성과 타깃 설정 (예: 날씨 정보로 전력 사용량 예측)
X_new = comb_df[['Temp_max', 'Temp_min', 'Dew_max', 'Precipit']]
y_new = comb_df['kWh_usage'].values

# Train/Test split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.3, random_state=0)

# --- Linear Regression 모델 ---
lr = LinearRegression()
lr.fit(X_train_new, y_train_new)
y_pred_lr_new = lr.predict(X_test_new)
print('Linear Regression Intercept:', lr.intercept_)
print('Linear Regression Coefficients:', lr.coef_)
df_linear = evaluation('Linear Regression', y_pred_lr_new, y_test_new)
print(df_linear)

# --- SVR Pipeline 모델 ---
svr = SVR(C=11, epsilon=1, kernel='rbf', gamma=3, tol=0.001, verbose=0)
pipe = Pipeline([
    ('StandardScaler', StandardScaler()),
    ('SVR', svr)
])
pipe.fit(X_train_new, y_train_new)
y_pred_svr = pipe.predict(X_test_new)
df_svr = evaluation('SVR', y_pred_svr, y_test_new)
print(df_svr)

# 샘플 예측 확인
print("Sample Test Inputs (first 10):")
print(X_test_new.head(10))
print("Linear Regression Predictions (first 10):")
print(lr.predict(X_test_new.head(10).values))
print("SVR Predictions (first 10):")
print(pipe.predict(X_test_new.head(10).values))
print("Actual kWh Usage (first 10):")
print(y_test_new[:10])

# 모델 저장 (joblib 사용)
joblib.dump(lr, './static/result/linear_regression_model.pkl')
print("Linear Regression model saved as './static/result/linear_regression_model.pkl'.")
joblib.dump(pipe, './static/result/svr_pipeline_model.pkl')
print("SVR Pipeline model saved as './static/result/svr_pipeline_model.pkl'.")

### 4. 평가 지표 및 예측 결과 플롯 ###
# (1) 평가 지표를 바 차트로 비교 플롯하기
metrics_df = pd.concat([df_linear, df_svr], axis=1)
plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar', rot=45)
plt.title("Evaluation Metrics Comparison")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig('./static/result/evaluation_metrics.png')
plt.show()

# (2) 실제 값과 예측 값의 산점도 플롯 (각 모델별)
plt.figure(figsize=(12, 5))

# Linear Regression 산점도
plt.subplot(1, 2, 1)
plt.scatter(y_test_new, y_pred_lr_new, alpha=0.7, color='blue')
plt.plot([min(y_test_new), max(y_test_new)], [min(y_test_new), max(y_test_new)], color='red', linestyle='--')
plt.xlabel("Actual kWh Usage")
plt.ylabel("Predicted kWh Usage")
plt.title("Linear Regression: Actual vs Predicted")

# SVR 산점도
plt.subplot(1, 2, 2)
plt.scatter(y_test_new, y_pred_svr, alpha=0.7, color='green')
plt.plot([min(y_test_new), max(y_test_new)], [min(y_test_new), max(y_test_new)], color='red', linestyle='--')
plt.xlabel("Actual kWh Usage")
plt.ylabel("Predicted kWh Usage")
plt.title("SVR: Actual vs Predicted")

plt.tight_layout()
plt.savefig('./static/result/actual_vs_predicted.png')
plt.show()