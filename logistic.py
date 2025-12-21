import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
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
df = pd.read_csv('mental_health_screen_time_dataset.csv')


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


def descriptive(df):
    df_count = df_clean.count()
    df_min = df_clean.min()
    df_max = df_clean.max()
    df_mean = df_clean.mean()
    df_median = df_clean.median()
    df_q1 = df_clean.quantile(0.25)
    df_q2 = df_clean.quantile(0.5)
    df_q3 = df_clean.quantile(0.75)
    df_iqr = df_q3 - df_q1
    df_varience = df_clean.var()
    df_stdev = df_clean.std()
    data = {
        "count": list(df_count),
        "Min" : list(df_min),
        "Max" : list(df_max),
        "Mean" : list(df_mean),
        "median" : list(df_median),
        "q1": list(df_q1),
        "q2":list(df_q2),
        "q3" : list(df_q3),
        "IQR": list(df_iqr),
        "Variencce": list(df_varience),
        "Stdev": list(df_stdev)
    }
    df_data =pd.DataFrame(data)
    df_data.index = df_clean.keys()
    df_complete = df_data.transpose()
    print(df_complete.to_string())
    return df_complete
descriptive(df_clean)


# Xóa giá trị ở cột Well_Being_Score lớn hơn 10 hoặc nhỏ hơn 0
df_clean = df_clean[~( (df_clean['Well_Being_Score'] > 10) | (df_clean['Well_Being_Score'] < 0) )]

# Xóa giá trị ở cột Stress_Level lớn hơn 10 hoặc nhỏ hơn 0
df_clean = df_clean[~( (df_clean['Stress_Level'] > 10) | (df_clean['Stress_Level'] < 0) )]

# Xóa giá trị ở cột Mood_Rating lớn hơn 10 hoặc nhỏ hơn 0
df_clean = df_clean[~( (df_clean['Mood_Rating'] > 10) | (df_clean['Mood_Rating'] < 0) )]

print(f"Kích thước của df_clean sau khi lọc các giá trị không hợp lệ:: {df_clean.shape}")


df_clean.describe()


def draw_box_plot(df):
    plt.figure(figsize=(10, 6))
    plt.boxplot(df.values, patch_artist=True)
    plt.xticks(range(1, len(df.columns) + 1), df.columns, rotation=45)
    plt.title("Boxplot of Numeric Columns", fontweight='bold')
    plt.grid(linestyle='solid', linewidth=0.4)
    plt.tight_layout()
    plt.show()
draw_box_plot(df_clean)


def draw_histogram(df):
    df_filtered = df.drop(columns=['ID', 'Mood_Rating'], errors='ignore')

    df_filtered.hist(bins=40,figsize=(15, 8),grid=True,edgecolor='black')
    plt.suptitle("Histogram of Numeric Columns",fontweight='bold',fontsize=16)
    plt.show()

draw_histogram(df_clean)


def draw_heatmap(df):
    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Heatmap of Correlation Matrix", fontweight='bold', fontsize=14)
    plt.show()
draw_heatmap(df_clean)


def draw_binned_stress_probability(df):
    df_tmp = df_clean.copy()

    df_tmp['High_Stress'] = df_tmp['Stress_Level'] >= 6
    df_tmp['Time_Bin'] = pd.qcut(df_tmp['App_Social_Media_Time'], q=6)

    stress_rate = df_tmp.groupby(
        'Time_Bin',
        observed=True
    )['High_Stress'].mean()

    plt.figure(figsize=(10, 6))
    stress_rate.plot(marker='o')
    plt.ylabel('Probability of High Stress')
    plt.xlabel('Social Media Time (Binned)')
    plt.title('Probability of High Stress vs Social Media Time', fontweight='bold')
    plt.grid(linestyle='solid', linewidth=0.4)
    plt.show()
draw_binned_stress_probability(df_clean)


def draw_boxplot_by_group(df_clean, features, group='Health_Status'):
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=group, y=feature, data=df)

        # Đổi nhãn nếu nhóm là Health_Status hoặc High_Stress
        if group == 'Health_Status':
            plt.xticks([0, 1], ['Poor Health', 'Good Health'])

        plt.title(f'{feature} vs {group}', fontweight='bold')
        plt.grid(linestyle='solid', linewidth=0.4)
        plt.show()

df['Health_Status'] = df['Well_Being_Score'] >= 7
df['Health_Status'] = df['Health_Status'].map({True:1, False:0})

features = ['App_Work_Time','App_Entertainment_Time','Sleep_Duration']
# Vẽ theo nhóm Health_Status
draw_boxplot_by_group(df_clean, features, group='Health_Status')


def draw_scatter(df_clean, x_feature, y_feature, hue=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_feature, y=y_feature, hue=hue, data=df, alpha=0.6)
    plt.title(f'{y_feature} vs {x_feature}', fontweight='bold')
    plt.grid(linestyle='solid', linewidth=0.4)
    plt.show()

df['High_Stress'] = df['Stress_Level'] >= 6

# Ví dụ Scatter giữa thời gian sử dụng và stress / mood
draw_scatter(df_clean, 'Daily_Screen_Time', 'Stress_Level', hue='High_Stress')
draw_scatter(df_clean, 'Sleep_Duration', 'Well_Being_Score', hue='Health_Status')


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


# Visualize data after outlier removal
def draw_box_plot(df):
    plt.figure(figsize=(10, 6))
    plt.boxplot(df.values, patch_artist=True)
    plt.xticks(range(1, len(df.columns) + 1), df.columns, rotation=45)
    plt.title("Boxplot of Numeric Columns (After Outlier Removal)", fontweight='bold')
    plt.grid(linestyle='solid', linewidth=0.4)
    plt.tight_layout()
    plt.show()
draw_box_plot(df_clean)


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


# ĐÁNH GIÁ HOLD-OUT
# Dự đoán xác suất lớp 1 (Stress cao)
y_prob_test = holdout_model.predict_proba(X_test)[:, 1]

# Chuyển xác suất sang nhãn nhị phân với ngưỡng 0.5
y_pred_test = (y_prob_test >= 0.5).astype(int)

# Các chỉ số đánh giá
holdout_auc = roc_auc_score(y_test, y_prob_test)
holdout_loss = log_loss(y_test, y_prob_test)
holdout_f1 = f1_score(y_test, y_pred_test)

holdout_acc = accuracy_score(y_test, y_pred_test)
holdout_precision = precision_score(y_test, y_pred_test)
holdout_recall = recall_score(y_test, y_pred_test)
holdout_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
holdout_mcc = matthews_corrcoef(y_test, y_pred_test)

holdout_pr_auc = average_precision_score(y_test, y_prob_test)
holdout_brier = brier_score_loss(y_test, y_prob_test)

# In kết quả
print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM TRA (HOLD-OUT)")
print(f"AUC: {holdout_auc:.3f}")
print(f"Log Loss: {holdout_loss:.3f}")
print(f"F1-score: {holdout_f1:.3f}")
print(f"Accuracy: {holdout_acc:.3f}")
print(f"Precision: {holdout_precision:.3f}")
print(f"Recall: {holdout_recall:.3f}")
print(f"Balanced Accuracy: {holdout_bal_acc:.3f}")
print(f"MCC: {holdout_mcc:.3f}")
print(f"PR-AUC: {holdout_pr_auc:.3f}")
print(f"Brier Score: {holdout_brier:.3f}")


# TỔNG HỢP KẾT QUẢ K-FOLD

kf_df = pd.DataFrame(kf_results)

print("\nKẾT QUẢ ĐÁNH GIÁ THEO STRATIFIED K-FOLD")
print(kf_df)

print("\nGIÁ TRỊ TRUNG BÌNH K-FOLD")
print(kf_df.mean(numeric_only=True))

print("\nĐỘ LỆCH CHUẨN K-FOLD")
print(kf_df.std(numeric_only=True))


print("SO SÁNH")
print(f"AUC Hold-out: {holdout_auc:.3f}")
print(f"AUC trung bình K-Fold: {kf_df['AUC'].mean():.3f}")


# Giả sử y_test và y_prob_test là kết quả từ Model
y_pred_test = (y_prob_test >= 0.5).astype(int)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Mô hình dự đoán tổng quát")
plt.show()


#ROC Curve
roc_disp = RocCurveDisplay.from_predictions(y_test, y_prob_test)
plt.title("ROC Curve - Mô hình dự đoán tổng quát")
plt.show()


df1 = df_clean.copy()

#DEFINE EXTREME STRESS LABEL
sleep_q25 = df1['Sleep_Duration'].quantile(0.25)
screen_q75 = df1['Daily_Screen_Time'].quantile(0.75)

df1['Extreme_Stress'] = (
    (df1['Stress_Level'] >= 8) &
    (
        (df1['Sleep_Duration'] <= sleep_q25) |
        (df1['Daily_Screen_Time'] >= screen_q75)
    )
).astype(int)

print(df1['Extreme_Stress'].value_counts())

#FEATURE ENGINEERING (SELECTIVE)
#Log-transform (chỉ các biến lệch mạnh)
df1['Daily_Screen_Time_log'] = np.log1p(df1['Daily_Screen_Time'])
df1['Phone_Unlocks_log'] = np.log1p(df1['Phone_Unlocks'])

#Ratio
df1['Screen_Sleep_Ratio'] = df['Daily_Screen_Time'] / df1['Sleep_Duration']

# Interaction
df1['Screen_x_Sleep'] = df1['Daily_Screen_Time'] * df1['Sleep_Duration']


#4. CLEAN INF / NAN
df1.replace([np.inf, -np.inf], np.nan, inplace=True)
df1.dropna(inplace=True)

#FEATURE SELECTION
features = [
    'Daily_Screen_Time_log',
    'Phone_Unlocks_log',
    'Sleep_Duration',
    'Screen_Sleep_Ratio',
    'Screen_x_Sleep'
]

X = df1[features]
y = df1['Extreme_Stress']

#TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
#SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#LOGISTIC REGRESSION
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

#Đánh giá
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Extreme Stress Detection (Log + Interaction) ===")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))


odds = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0],
    'Odds_Ratio': np.exp(model.coef_[0])
}).sort_values(by='Odds_Ratio', ascending=False)

print(odds)


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Extreme Stress Detection')
plt.legend()
plt.grid(True)
plt.show()


import shap

explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=features
)

