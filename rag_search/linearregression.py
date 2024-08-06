import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('Real_estate.csv')

# 设置X和y
X = df.drop(['Y house price of unit area'], axis=1)
y = df['Y house price of unit area']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# 初始化模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("均方误差 (MSE):", mse)
print("R² 分数:", r2)

# 可视化预测与实际值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('实际房价')
plt.ylabel('预测房价')
plt.title('实际房价 vs 预测房价')
plt.show()

# 用户输入特征
def get_user_input(features):
    user_input = []
    for feature in features:
        value = float(input(f"请输入 {feature} 的值: "))
        user_input.append(value)
    return np.array([user_input])

# 输入特征
input_features = X.columns.tolist()
user_input = get_user_input(input_features)

# 预测
predicted_price = model.predict(user_input)
print(f"预测的房价为: {predicted_price[0]:.2f} 单位面积价格")