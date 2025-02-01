# sampling.py

import numpy as np
import joblib

class SklearnSampler:
    def __init__(self, model):
        """
        Args:
            model: scikit-learn 모델 (예: LinearRegression 또는 SVR Pipeline)
        """
        self.model = model

    def predict(self, input_features):
        """
        Args:
            input_features (list or np.ndarray): [Temp_max, Temp_min, Dew_max, Precipit]
               4개 피처를 가진 1차원 배열.
        Returns:
            float: 모델 예측값.
        """
        # 입력을 (1, feature_dim) 모양으로 변환 후 예측 수행
        input_array = np.array(input_features).reshape(1, -1)
        prediction = self.model.predict(input_array)[0]
        return prediction

if __name__ == "__main__":
    # 저장된 모델 로드
    lr_model = joblib.load('./static/result/linear_regression_model.pkl')
    svr_model = joblib.load('./static/result/svr_pipeline_model.pkl')

    # 두 모델에 대해 샘플러 인스턴스 생성
    lr_sampler = SklearnSampler(lr_model)
    svr_sampler = SklearnSampler(svr_model)

    # 예시 입력: [Temp_max, Temp_min, Dew_max, Precipit]
    # (필요에 따라 적절한 값을 입력하세요)
    sample_input = [85, 70, 65, 0.0]

    # 각 모델로 예측 수행
    pred_lr = lr_sampler.predict(sample_input)
    pred_svr = svr_sampler.predict(sample_input)

    print("Linear Regression Prediction:", pred_lr)
    print("SVR Prediction:", pred_svr)
