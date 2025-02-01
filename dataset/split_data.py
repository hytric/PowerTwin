import pandas as pd

def split_power_data_by_year(input_csv):
    # 1) CSV 로드, StartDate 열을 datetime으로 파싱
    df = pd.read_csv(input_csv, parse_dates=["StartDate"])
    
    # 2) 2020년 기준 분리
    #  - StartDate >= '2020-01-01' 인 행을 test set
    #  - StartDate <  '2020-01-01' 인 행을 train set
    df_train = df[df["StartDate"] < "2020-01-01"]
    df_test  = df[df["StartDate"] >= "2020-01-01"]

    # 3) 각각 저장
    df_train.to_csv("./train/power.csv", index=False)
    df_test.to_csv("./test/power.csv", index=False)

    print("Split power completed.")
    print(f"Train power dataset size: {len(df_train)}")
    print(f"Test power dataset size:  {len(df_test)}")

def split_wether_data_by_year(input_csv):
    # 1) CSV 로드, StartDate 열을 datetime으로 파싱
    df = pd.read_csv(input_csv, parse_dates=["Date"])
    
    # 2) 2020년 기준 분리
    #  - Date >= '2020-01-01' 인 행을 test set
    #  - Date <  '2020-01-01' 인 행을 train set
    df_train = df[df["Date"] < "2020-01-01"]
    df_test  = df[df["Date"] >= "2020-01-01"]

    # 3) 각각 저장
    df_train.to_csv("./train/wether.csv", index=False)
    df_test.to_csv("./test/wether.csv", index=False)

    print("Split wether completed.")
    print(f"Train wether dataset size: {len(df_train)}")
    print(f"Test wether dataset size:  {len(df_test)}")


if __name__ == "__main__":
    # 원본 CSV 파일을 지정
    power_input_csv = "./power_usage_2016_to_2020.csv"  # 예) StartDate,Value (kWh),day_of_week,notes
    wether_input_csv = "./weather_2016_2020_daily.csv"  # 예) StartDate,Value (kWh),day_of_week,notes
    split_power_data_by_year(power_input_csv)
    split_wether_data_by_year(wether_input_csv)