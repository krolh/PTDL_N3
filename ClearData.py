from multiprocessing.reduction import duplicate

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

df =pd.read_csv("mental_health_screen_time_dataset.csv")
df.info()
df_numeric = df.select_dtypes(include=['number'])
df_clean =df_numeric.drop(columns=["Participant_ID"])

# #kiểm tra dữ liệu thiếu
# miss_value = df_clean.isnull().sum()
# miss_precent =(miss_value / len(df_clean))*100
# miss_df = pd.DataFrame({
#     'Miss Values': miss_value,
#     'Miss precent' : miss_precent
# })
# print(miss_df)

# Kiểm tra dữ liệu trùng lặp
duplicates = df_clean.duplicated().sum()
duplicates_rows = df[df_clean.duplicated(keep=False)]

print("Số dòng trùng lặp: ", duplicates)
print("\nDòng trùng lặp:\n ", duplicates_rows)
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
#kiểm tra dữ liệu thiếu
miss_value = df_clean.isnull().sum()
miss_precent =(miss_value / len(df_clean))*100
miss_df = pd.DataFrame({
    'Miss Values': miss_value,
    'Miss precent' : miss_precent
})
print(miss_df)
df_clean = df_clean.drop_duplicates()
print(df_clean.shape)