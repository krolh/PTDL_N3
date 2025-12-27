import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import  joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score,
    accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef,
    average_precision_score, brier_score_loss,
    classification_report
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


#Đọc dữ liệu
df = pd.read_csv("mental_health_screen_time_dataset.csv")


df.info()
df_numeric = df.select_dtypes(include=['number'])
df_clean =df_numeric.drop(columns=["Participant_ID"])

#kiểm tra dữ liệu thiếu
miss_value = df_clean.isnull().sum()
miss_precent =(miss_value / len(df_clean))*100
miss_df = pd.DataFrame({
    'Miss Values': miss_value,
    'Miss precent' : miss_precent
})
print(miss_df)


#kiểm tra dữ liệu thiếu
miss_value = df_clean.isnull().sum()
miss_precent =(miss_value / len(df_clean))*100
miss_df = pd.DataFrame({
    'Miss Values': miss_value,
    'Miss precent' : miss_precent
})
print(miss_df)


# điền các giá trị khuyết
for column in df_clean.columns:
    # Kiểm tra xem cột có phải là kiểu dữ liệu số hay không (float, int)
    if np.issubdtype(df_clean[column].dtype, np.number):
        # Tính giá trị trung bình của cột (bỏ qua các giá trị null)
        mean_value = df_clean[column].mean()
        # Thay thế các giá trị null bằng giá trị trung bình vừa tính
        df_clean[column].fillna(mean_value, inplace=True)
        print(f"-> Đã xử lý cột số '{column}': Thay thế NaN bằng giá trị trung bình ({mean_value:.2f}).")

    elif df_clean[column].dtype == 'object' or df_clean[column].dtype == 'category':

        pass

df_clean.info()


# Kiểm tra dữ liệu trùng lặp
duplicates = df_clean.duplicated().sum()
duplicates_rows = df[df_clean.duplicated(keep=False)]

print("Số dòng trùng lặp: ", duplicates)
print("\nDòng trùng lặp:\n ", duplicates_rows)


#Xóa dữ liệu trùng
df_clean = df_clean.drop_duplicates()
print(df_clean.shape)


#Xóa dữ liệu trùng
df_clean = df_clean.drop_duplicates()
print(df_clean.shape)


#Đổi giờ sang phút
df_clean['Sleep_Duration'] = df_clean['Sleep_Duration']*60
print(df_clean['Sleep_Duration'].head(10))

df_processed = df_clean.copy()
#Định nghĩa các nhóm cột theo cách xử lý
capping_cols = ['Daily_Screen_Time', 'Phone_Unlocks']
drop_cols = ['App_Work_Time', 'App_Entertainment_Time', 'App_Social_Media_Time']
#Hàm xử lý Outlier
def handle_outliers(df, columns, method='capping'):
    for col in columns:
        # Tính toán IQR cho từng cột cụ thể
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if method == 'capping':
            df[col] = df[col].clip(lower, upper)
        elif method == 'drop':
            # Lọc dữ liệu lấy các dòng nằm trong khoảng an toàn
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df
#Capping cho nhóm Daily_Screen_Time và Phone_Unlocks
df_processed = handle_outliers(df_processed, capping_cols, method='capping')
#Xóa dòng cho nhóm các ứng dụng
df_processed = handle_outliers(df_processed, drop_cols, method='drop')
#Kiểm tra kết quả
print(f"Kích thước ban đầu: {df_clean.shape}")
print(f"Kích thước sau khi xử lý Outliers: {df_processed.shape}")
# Cập nhật lại df_clean
df_clean = df_processed.copy()
df = df_clean.copy()

# Feature Engineering
# Log-transform các biến hành vi
log_cols = [
    'Daily_Screen_Time',
    'App_Social_Media_Time',
    'App_Work_Time',
    'App_Entertainment_Time',
    'Phone_Unlocks'
]

for col in log_cols:
    df[col + '_log'] = np.log1p(df[col])

# Tạo các biến tỷ lệ
df['Social_Ratio'] = df['App_Social_Media_Time'] / df['Daily_Screen_Time']
df['Entertainment_Ratio'] = df['App_Entertainment_Time'] / df['Daily_Screen_Time']
df['Work_Ratio'] = df['App_Work_Time'] / df['Daily_Screen_Time']

#Xử lý chia cho 0 và giá trị thiếu
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

#Tạo các biến tương tác
df['Screen_x_Sleep'] = df['Daily_Screen_Time'] * df['Sleep_Duration']
df['Unlocks_x_Social'] = df['Phone_Unlocks'] * df['App_Social_Media_Time']

#Tạo biến mục tiêu
df['High_Stress'] = (df['Stress_Level'] >= 6).astype(int)

print("\n: Hoàn thành")


#Feature Selection
feature_cols = [
    'Daily_Screen_Time_log',
    'App_Social_Media_Time_log',
    'App_Work_Time_log',
    'App_Entertainment_Time_log',
    'Phone_Unlocks_log',
    'Social_Ratio',
    'Entertainment_Ratio',
    'Work_Ratio',
    'Screen_x_Sleep',
    'Unlocks_x_Social',
    'Sleep_Duration'
]

X = df[feature_cols]
y = df['High_Stress']

print("Số lượng đặc trưng:", X.shape[1])


#Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Chia tập
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"Số mẫu huấn luyện: {X_train.shape[0]}")
print(f"Số mẫu kiểm tra: {X_test.shape[0]}")


#Xây dựng mô hình
model_config = {
    "penalty": "elasticnet",
    "solver": "saga",
    "l1_ratio": 0.5,
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": 42
}

#print("Các siêu tham số:", model_config)


#Huấn luyện theo phương pháp Hold-out
holdout_model = LogisticRegression(**model_config)
holdout_model.fit(X_train, y_train)


#Huấn luyện với Stratified K-Fold Cross Validation
#Khởi tạo Stratified K-Fold với 5 fold
kfold = StratifiedKFold(
    n_splits=5,        #Số lượng fold
    shuffle=True,      #Trộn dữ liệu trước khi chia
    random_state=42    #Cố định seed để đảm bảo tái lập kết quả
)

print("Bắt đầu huấn luyện với Stratified K-Fold")

kf_results = []

fold = 1  # Biến đếm số thứ tự fold

# Vòng lặp qua từng fold
for train_idx, val_idx in kfold.split(X_train, y_train):

    print(f"\n--- Fold {fold} ---")

    # X_tr, y_tr: dữ liệu dùng để huấn luyện mô hình trong fold hiện tại
    # X_val, y_val: dữ liệu kiểm tra nội bộ (validation) trong fold hiện tại
    X_tr = X_train[train_idx]
    y_tr = y_train.iloc[train_idx]

    X_val = X_train[val_idx]
    y_val = y_train.iloc[val_idx]

    # Khởi tạo mô hình Logistic Regression mới cho mỗi fold
    # Việc khởi tạo lại đảm bảo các fold là độc lập, không bị rò rỉ dữ liệu
    model = LogisticRegression(**model_config)

    # Huấn luyện mô hình trên tập huấn luyện của fold hiện tại
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    kf_results.append({
        "AUC": roc_auc_score(y_val, y_prob),
        "LogLoss": log_loss(y_val, y_prob),
        "F1": f1_score(y_val, y_pred)
    })

    fold += 1

print("Hoàn tất quá trình huấn luyện với Stratified K-Fold")


# 8. Lưu model & scaler
# ======================
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🎉 Đã lưu logistic_model.pkl và scaler.pkl")
