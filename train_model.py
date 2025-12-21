import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ======================
# 1. Đọc dữ liệu
# ======================
df = pd.read_csv("mental_health_screen_time_dataset_clean.csv")

print("📌 Kích thước dữ liệu:", df.shape)
print("📌 Các cột trong dataset:")
print(df.columns)

# ======================
# 2. Chọn đặc trưng (4 CỘT)
# ======================
features = [
    "Daily_Screen_Time",
    "App_Social_Media_Time",
    "Sleep_Duration",
    "Phone_Unlocks"
]

target = "Well_Being_Score"

X = df[features]
y = df[target]

# ======================
# 3. Chuyển nhãn về nhị phân
# ======================
# >= 5: ổn định (0)
# < 5: bị ảnh hưởng (1)
y = (y < 5).astype(int)

# ======================
# 4. Chia tập train / test
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 5. Chuẩn hóa dữ liệu
# ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("📌 Scaler fit với số features:", X_train.shape[1])

# ======================
# 6. Train model
# ======================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ======================
# 7. Đánh giá
# ======================
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# ======================
# 8. Lưu model & scaler
# ======================
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🎉 Đã lưu logistic_model.pkl và scaler.pkl")
