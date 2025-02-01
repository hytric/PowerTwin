# sand_data.py

import time
import requests
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
from sampling import SklearnSampler
import matplotlib.pyplot as plt
import os

# Ditto config
USERNAME = "ditto"
PASSWORD = "ditto"
DITTO_BASE_URL = "http://localhost:8080/api/2"
THING_ID = "mycompany:device01"
FEATURE_PREFIX = "sensor_"  # 날짜별 Feature를 위한 접두사

POWER_CSV = "./dataset/test/power.csv"
WEATHER_CSV = "./dataset/test/weather.csv"


def reset_ditto_thing():
    """
    Ditto에서 기존 Thing(mycompany:device01)을 삭제하고, 다시 생성하여 초기화합니다.
    (정책은 "mycompany:device01"으로 연결되어 있다고 가정)
    """
    url = f"{DITTO_BASE_URL}/things/{THING_ID}"
    resp = requests.delete(url, auth=(USERNAME, PASSWORD))
    if resp.ok:
        print(f"[OK] Deleted Thing {THING_ID} successfully.")
    else:
        print(f"[WARN] {resp.status_code} {resp.text} while deleting Thing {THING_ID}.")

    # Thing 생성
    resp = requests.put(url, auth=(USERNAME, PASSWORD), json={"policyId": "mycompany:device01"})
    if resp.ok:
        print(f"[OK] Created Thing {THING_ID} successfully.")
    else:
        print(f"[ERR] {resp.status_code} {resp.text} while creating Thing {THING_ID}.")


def get_feature_properties(feature_id):
    """
    특정 Feature의 properties를 GET으로 가져옵니다.
    존재하지 않을 경우 빈 dict를 반환합니다.
    """
    url = f"{DITTO_BASE_URL}/things/{THING_ID}/features/{feature_id}/properties"
    resp = requests.get(url, auth=(USERNAME, PASSWORD))
    return resp.json() if resp.ok else {}


def put_feature_properties(feature_id, new_properties):
    """
    특정 Feature의 properties를 PUT으로 저장합니다.
    """
    url = f"{DITTO_BASE_URL}/things/{THING_ID}/features/{feature_id}/properties"
    resp = requests.put(url, auth=(USERNAME, PASSWORD), json=new_properties)
    if resp.ok:
        print(f"[OK] Updated properties for {feature_id} successfully.")
    else:
        print(f"[ERR] {resp.status_code} {resp.text} for feature {feature_id}")


def update_feature(feature_id, daily_data, hourly_data):
    """
    (1) 기존 properties를 GET한 후 dailyData와 hourlyData를 갱신하고 PUT으로 업데이트합니다.
    
    Args:
        feature_id (str): Feature ID.
        daily_data (dict): 날짜 단위 정보.
        hourly_data (dict): 시간 단위 정보 (새로운 기록).
    """
    current_props = get_feature_properties(feature_id)
    if not current_props:
        # 초기 구조 설정
        current_props = {"dailyData": {}, "hourlyData": []}

    if daily_data:
        current_props["dailyData"].update(daily_data)
    if hourly_data:
        current_props["hourlyData"].append(hourly_data)

    put_feature_properties(feature_id, current_props)


def get_feature_data(feature_id):
    """
    특정 Feature의 전체 데이터를 Ditto에서 가져옵니다.
    (dailyData와 hourlyData 전부)
    """
    current_props = get_feature_properties(feature_id)
    if not current_props:
        return {"dailyData": {}, "hourlyData": []}
    return current_props


def extract_daily_weather(w_row):
    """
    주어진 weather row에서 필요한 날씨 피처들을 추출하여 dict로 반환합니다.
    """
    keys = ["day_of_week", "Temp_max", "Temp_min", "Temp_avg",
            "Dew_max", "Dew_min", "Dew_avg",
            "Hum_max", "Hum_min", "Hum_avg",
            "Wind_max", "Wind_min", "Wind_avg",
            "Press_max", "Press_min", "Press_avg",
            "Precipit"]
    return {k: float(w_row.get(k, 0.0)) for k in keys}


def ensure_feature_exists(feature_id):
    """
    Ditto에 해당 feature가 존재하는지 확인하고, 없으면 생성합니다.
    """
    if not get_feature_properties(feature_id):
        url = f"{DITTO_BASE_URL}/things/{THING_ID}/features/{feature_id}"
        resp = requests.put(url, auth=(USERNAME, PASSWORD), json={"properties": {}})
        if resp.ok:
            print(f"[OK] Created feature {feature_id}.")
        else:
            print(f"[ERR] {resp.status_code} {resp.text} while creating feature {feature_id}.")


def run_model_analysis(date_data, lr_sampler, svr_sampler):
    """
    주어진 date_data (dailyData, hourlyData)를 사용하여 두 모델(lr_sampler, svr_sampler)로 예측을 수행합니다.
    만약 date_data에 'lr_prediction'과 'svr_prediction' 키가 이미 존재하면 해당 값을 사용하고,
    없으면 모델 예측 후 date_data에 추가합니다.
    
    Args:
        date_data (dict): 'dailyData' 및 'hourlyData'를 포함하는 사전.
        lr_sampler: Linear Regression 모델의 sampler 객체.
        svr_sampler: SVR Pipeline 모델의 sampler 객체.
        
    Returns:
        tuple: (lr_prediction, svr_prediction)
    """
    daily_data = date_data.get("dailyData", {})

    # 사용될 피처: Temp_max, Temp_min, Dew_max, Precipit
    weather_features = [
        float(daily_data.get("Temp_max", 0.0)),
        float(daily_data.get("Temp_min", 0.0)),
        float(daily_data.get("Dew_max", 0.0)),
        float(daily_data.get("Precipit", 0.0)),
    ]
    w_feats = np.array(weather_features, dtype=float)

    # 이미 예측값이 있다면 재사용
    lr_pred = date_data.get("lr_prediction")
    svr_pred = date_data.get("svr_prediction")

    if lr_pred is None or svr_pred is None:
        lr_pred = lr_sampler.predict(w_feats)
        svr_pred = svr_sampler.predict(w_feats)
        date_data["lr_prediction"] = lr_pred
        date_data["svr_prediction"] = svr_pred

    return lr_pred, svr_pred


def main():
    # [A] Ditto 초기화
    reset_ditto_thing()

    # [B] CSV 파일 로드 및 timestamp 변환
    df_power = pd.read_csv(POWER_CSV)
    df_weather = pd.read_csv(WEATHER_CSV)
    df_power["timestamp"] = pd.to_datetime(df_power["StartDate"])
    df_weather["timestamp"] = pd.to_datetime(df_weather["Date"])

    # [C] 저장된 모델 로드 및 Sampler 인스턴스 생성
    lr_model = joblib.load('./static/result/linear_regression_model.pkl')
    svr_model = joblib.load('./static/result/svr_pipeline_model.pkl')
    lr_sampler = SklearnSampler(lr_model)
    svr_sampler = SklearnSampler(svr_model)

    # [D] 일정 간격(테스트를 위해 interval=2초)으로 power 데이터를 순차 전송
    interval = 2  # 실제 운영 시에는 600초(10분) 등 적절하게 설정
    next_run = time.time()

    for i, row in df_power.iterrows():
        now = time.time()
        if now < next_run:
            time.sleep(next_run - now)
        next_run += interval

        # (1) 현재 row의 날짜와 feature_id 생성
        date_str = row["timestamp"].strftime("%Y-%m-%d")
        feature_id = f"{FEATURE_PREFIX}{date_str}"

        # (2) 해당 날짜의 weather 정보 추출
        w = df_weather[df_weather["timestamp"].dt.date == row["timestamp"].date()]
        if not w.empty:
            daily_data = extract_daily_weather(w.iloc[0])
        else:
            daily_data = {}

        # (3) 시간 단위 데이터 구성
        hourly_data = {
            "timestamp": row["timestamp"].strftime("%H:%M:%S"),
            "Value_kWh": float(row["Value (kWh)"]),
            "day_of_week": float(row.get("day_of_week", 0.0))
        }

        # (4) Feature 존재 여부 확인 및 생성
        ensure_feature_exists(feature_id)

        # (5) Feature 업데이트: dailyData 갱신 및 hourlyData 추가
        update_feature(feature_id, daily_data, hourly_data)
        print(f"Sent row {i} at {time.strftime('%Y-%m-%d %H:%M:%S')} for {date_str}")

        # (6) Ditto에 저장된 해당 날짜의 전체 data 가져오기
        date_data = get_feature_data(feature_id)

        # (7) 모델 분석 수행 (예측값이 없으면 예측 후 추가)
        lr_prediction, svr_prediction = run_model_analysis(date_data, lr_sampler, svr_sampler)

        # (8) 분석 결과를 dailyData에 추가하여 최종 업데이트
        if date_data.get("dailyData"):
            # dailyData가 dict이면 직접 key에 저장
            date_data["dailyData"]["lr_prediction"] = float(lr_prediction)
            date_data["dailyData"]["svr_prediction"] = float(svr_prediction)
        else:
            print(f"[WARN] dailyData is empty for {feature_id}!")

        put_feature_properties(feature_id, date_data)
        print(f"[OK] Updated analysis result for {date_str}.")

    print("[DONE] All rows sent and analyzed.")


if __name__ == "__main__":
    main()