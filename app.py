from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model & scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    chart_path = None

    if request.method == "POST":
        try:
            screen = float(request.form["screen_time"])
            social = float(request.form["social_time"])
            sleep = float(request.form["sleep_time"])
            age = float(request.form["age"])

            # Input
            X = np.array([[screen, social, sleep, age]])
            X_scaled = scaler.transform(X)

            # Predict
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1] * 100

            probability = round(prob, 2)
            result = "⚠ Có nguy cơ bị ảnh hưởng" if pred == 1 else "✅ Ít nguy cơ bị ảnh hưởng"

            # ====== VẼ BIỂU ĐỒ ======
            features = ["Screen", "Social", "Sleep", "Age"]
            values = [screen, social, sleep, age]

            fig, ax1 = plt.subplots(figsize=(8, 4))

            # CỘT GIÁ TRỊ ĐẦU VÀO
            ax1.bar(features, values)
            ax1.set_ylabel("Giá trị đầu vào (giờ / tuổi)")
            ax1.set_ylim(0, max(values) + 2)

            # TRỤC PHỤ - % DỰ ĐOÁN
            ax2 = ax1.twinx()
            ax2.plot(
                features,
                [probability] * len(features),
                color="red",
                linewidth=2,
                marker="o",
                label="Xác suất ảnh hưởng (%)"
            )
            ax2.set_ylabel("Xác suất ảnh hưởng (%)")
            ax2.set_ylim(0, 100)

            plt.title("Biểu đồ dữ liệu đầu vào & xác suất dự đoán")
            ax2.legend(loc="upper right")
            plt.tight_layout()

            chart_path = "static/prediction_chart.png"
            plt.savefig(chart_path)
            plt.close()

        except Exception as e:
            result = "❌ Dữ liệu nhập không hợp lệ"

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        chart_path=chart_path
    )

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
