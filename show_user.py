# show_user.py

from flask import Flask, jsonify, render_template_string, url_for
import requests
import re

# Ditto config
USERNAME = "ditto"
PASSWORD = "ditto"
DITTO_BASE_URL = "http://localhost:8080/api/2"
THING_ID = "mycompany:device01"

app = Flask(__name__)

def get_ditto_features():
    """
    Ditto에 저장된 모든 Features 목록을 가져옵니다.
    """
    url = f"{DITTO_BASE_URL}/things/{THING_ID}/features"
    resp = requests.get(url, auth=(USERNAME, PASSWORD))
    if resp.ok:
        return resp.json()
    return {}

@app.route("/api/dates")
def api_dates():
    """
    Ditto에서 날짜별 sensor Feature 목록을 추출하여 반환합니다.
    예: ["2024-02-01", "2024-02-02", ...]
    """
    features = get_ditto_features()
    date_list = []
    pattern = re.compile(r"sensor_(\d{4}-\d{2}-\d{2})")
    for feature in features.keys():
        match = pattern.match(feature)
        if match:
            date_list.append(match.group(1))
    date_list.sort()
    return jsonify(date_list)

@app.route("/api/date/<date_str>")
def api_date(date_str):
    """
    특정 날짜의 sensor_<date_str> Feature 데이터를 가져옵니다.
    반환 예시:
    {
      "dailyData": { ... },
      "hourlyData": [ {...}, {...}, ... ]
    }
    """
    feature_id = f"sensor_{date_str}"
    url = f"{DITTO_BASE_URL}/things/{THING_ID}/features/{feature_id}/properties"
    resp = requests.get(url, auth=(USERNAME, PASSWORD))
    if resp.ok:
        data = resp.json()
        return jsonify({
            "dailyData": data.get("dailyData", {}),
            "hourlyData": data.get("hourlyData", [])
        })
    return jsonify({"error": "Data not found"}), 404

@app.route("/")
def index():
    """
    메인 UI 페이지
    - 날짜 선택 드롭다운 및 "Show Chart" 버튼
    - 선택한 날짜의 dailyData를 테이블로 표시
    - 누적 Value_kWh와 모델 예측값(lr_prediction, svr_prediction)을 한 그래프에 표시
    - 분석 결과 이미지 (pairplot, heatmap, evaluation_metrics, actual_vs_predicted)를 제목, 캡션과 함께 삽입
    """
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Ditto Data Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { font-family: Arial, sans-serif; }
      .image-section { margin-top: 40px; }
      .image-container { margin-bottom: 30px; }
      .image-container img { max-width: 100%; height: auto; }
      .caption { font-size: 0.9em; color: #555; }
    </style>
</head>
<body>
    <h1>Ditto TimeSeries Visualization</h1>

    <label for="dateSelect">Select Date:</label>
    <select id="dateSelect"></select>
    <button onclick="fetchAndDraw()">Show Chart</button>

    <h2>Daily Data</h2>
    <table border="1" id="dailyTable">
        <thead><tr><th>Key</th><th>Value</th></tr></thead>
        <tbody></tbody>
    </table>

    <h2>Hourly Data & Model Analysis</h2>
    <canvas id="hourlyChart" width="800" height="400"></canvas>

    <!-- 이미지 섹션 -->
    <div class="image-section">
      <h2>Analysis Images</h2>
      
      <div class="image-container">
        <h3>Pair Plot</h3>
        <img src="{{ url_for('static', filename='result/pairplot.png') }}" alt="Pair Plot">
        <p class="caption">각 변수 간 분포와 산점도가 표시된 Pair Plot입니다.</p>
      </div>

      <div class="image-container">
        <h3>Heatmap</h3>
        <img src="{{ url_for('static', filename='result/heatmap.png') }}" alt="Heatmap">
        <p class="caption">수치형 변수 간 상관계수를 시각화한 Heatmap입니다.</p>
      </div>

      <div class="image-container">
        <h3>Evaluation Metrics</h3>
        <img src="{{ url_for('static', filename='result/evaluation_metrics.png') }}" alt="Evaluation Metrics">
        <p class="caption">모델 평가 지표 (MSE, MAE, R² 등)를 비교한 바 차트입니다.</p>
      </div>

      <div class="image-container">
        <h3>Actual vs Predicted</h3>
        <img src="{{ url_for('static', filename='result/actual_vs_predicted.png') }}" alt="Actual vs Predicted">
        <p class="caption">실제 전력 사용량과 예측값의 산점도를 보여주는 그래프입니다.</p>
      </div>
    </div>

    <script>
    window.addEventListener('DOMContentLoaded', () => {
        fetch('/api/dates')
          .then(res => res.json())
          .then(data => {
              const dateSelect = document.getElementById('dateSelect');
              data.forEach(d => {
                  const opt = document.createElement('option');
                  opt.value = d;
                  opt.textContent = d;
                  dateSelect.appendChild(opt);
              });
          })
          .catch(err => console.error(err));
    });

    function fetchAndDraw() {
        const dateSelect = document.getElementById('dateSelect');
        const selectedDate = dateSelect.value;
        if (!selectedDate) return;

        fetch('/api/date/' + selectedDate)
          .then(res => res.json())
          .then(data => {
              updateDailyTable(data.dailyData);
              drawHourlyChart(data.hourlyData, data.dailyData);
          })
          .catch(err => console.error(err));
    }

    function updateDailyTable(dailyData) {
        const tbody = document.querySelector('#dailyTable tbody');
        tbody.innerHTML = '';
        if(!dailyData) return;
        Object.keys(dailyData).forEach(key => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${key}</td><td>${dailyData[key]}</td>`;
            tbody.appendChild(row);
        });
    }

    let hourlyChart;
    function drawHourlyChart(hourlyData, dailyData) {
        if (hourlyChart) hourlyChart.destroy();

        const ctx = document.getElementById('hourlyChart').getContext('2d');
        const labels = hourlyData.map(d => d.timestamp);
        let cumulativeValue = 0;
        const cumulativeData = hourlyData.map(d => {
            cumulativeValue += parseFloat(d.Value_kWh);
            return parseFloat(cumulativeValue.toFixed(3));
        });

        // dailyData에 저장된 예측값을 사용
        const lr_pred = dailyData.lr_prediction !== undefined ? dailyData.lr_prediction : null;
        const svr_pred = dailyData.svr_prediction !== undefined ? dailyData.svr_prediction : null;
        const lrLine = lr_pred !== null ? new Array(labels.length).fill(lr_pred) : [];
        const svrLine = svr_pred !== null ? new Array(labels.length).fill(svr_pred) : [];

        hourlyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Cumulative Value_kWh',
                        data: cumulativeData,
                        borderColor: 'blue',
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Linear Regression Prediction',
                        data: lrLine,
                        borderColor: 'red',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'SVR Prediction',
                        data: svrLine,
                        borderColor: 'green',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Cumulative Value_kWh and Model Predictions'
                    }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }
    </script>
</body>
</html>
    """.strip()
    return render_template_string(html_content)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8085)