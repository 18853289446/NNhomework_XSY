import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =====================读取数据=====================
df = pd.read_csv("Concrete_Data_Yeh.csv")

# 特征与标签
X = df.drop("csMPa", axis=1).values
y = df["csMPa"].values

# =====================相关性分析=====================
corr = df.corr()
print("\n特征与抗压强度相关性：")
print(corr["csMPa"].sort_values(ascending=False))

# =====================划分训练集/测试集（8:2）=====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================标准化=====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 在特征前加一列 1，用于计算截距（偏置 b）
X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# =====================梯度下降线性回归=====================
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])  # 初始化权重 w 和偏置 b 为 0
    cost_history = []  # 记录损失变化

    for i in range(n_iterations):
        # 预测值h = X @ theta
        y_pred = X.dot(theta)
        
        # 误差
        error = y_pred - y
        
        # 梯度
        gradient = (1 / m) * X.T.dot(error)
        
        # 更新权重
        theta = theta - learning_rate * gradient
        
        # 记录MSE损失
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

    return theta, cost_history

# 训练模型
theta, cost_history = gradient_descent(X_train_b, y_train, learning_rate=0.01, n_iterations=1000)

# 提取参数
b = theta[0]  # 偏置
w = theta[1:] # 特征权重

# ===================== 预测与评估 =====================
y_pred = X_test_b.dot(theta)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== 梯度下降线性回归 模型评估 =====")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# 输出权重
print("\n特征权重（回归系数）：")
for name, coef in zip(df.drop("csMPa", axis=1).columns, w):
    print(f"{name:18s} : {coef:.4f}")

# ====================图像======================
# 图1：损失下降曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Cost History")
plt.grid(True)

# 图2：真实值 vs 预测值
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("True Compressive Strength")
plt.ylabel("Predicted Compressive Strength")
plt.title("True vs Predicted")
plt.grid(True)

plt.tight_layout()
plt.show()