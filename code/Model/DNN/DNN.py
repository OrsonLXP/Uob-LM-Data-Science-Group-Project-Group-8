import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取数据
file_path = "data/eTraffic_lag1.csv"
df = pd.read_csv(file_path)

# 选择特征和目标变量
features = df.drop(columns=["local_authority_name","year" ,"local_authority_name_encoded",
                            "GDP", "pedal_cycles", "two_wheeled_motor_vehicles",
                            "cars_and_taxis", "HGVs_3_rigid_axle", "HGVs_5_articulated_axle",
                            "all_HGVs", "all_motor_vehicles"])
target = df["GDP"].values  # 目标变量，确保是 1D 数组

# 归一化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 设置 K 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储各折的评估指标
train_mae_list, train_mse_list, train_r2_list, train_mape_list = [], [], [], []
test_mae_list, test_mse_list, test_r2_list, test_mape_list = [], [], [], []

# 定义模型构建函数
def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.01), input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),
        layers.Dense(1)  # 输出层
    ])

    # 使用 AdamW 优化器
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae"])
    return model

# 进行 K-Fold 训练与评估
for fold, (train_idx, test_idx) in enumerate(kf.split(features_scaled)):
    print(f"\nFold {fold+1}/{kf.get_n_splits()}")

    # 训练集和测试集划分
    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]

    # 构建模型
    model = build_model(X_train.shape[1])

    # 回调函数
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)

    # 训练模型
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr], verbose=0)

    # 评估模型（训练集）
    y_train_pred = model.predict(X_train).reshape(-1)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100  # 计算 MAPE

    # 评估模型（测试集）
    y_test_pred = model.predict(X_test).reshape(-1)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100  # 计算 MAPE

    # 记录指标
    train_mae_list.append(train_mae)
    train_mse_list.append(train_mse)
    train_r2_list.append(train_r2)
    train_mape_list.append(train_mape)

    test_mae_list.append(test_mae)
    test_mse_list.append(test_mse)
    test_r2_list.append(test_r2)
    test_mape_list.append(test_mape)

    # 输出当前折的结果
    print(f"Train - MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAPE: {train_mape:.2f}%")
    print(f"Test  - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAPE: {test_mape:.2f}%")

# 计算平均评估指标
print("\n=== Cross-Validation Results ===")
print(f"Train - MAE: {np.mean(train_mae_list):.4f}, MSE: {np.mean(train_mse_list):.4f}, R²: {np.mean(train_r2_list):.4f}, MAPE: {np.mean(train_mape_list):.2f}%")
print(f"Test  - MAE: {np.mean(test_mae_list):.4f}, MSE: {np.mean(test_mse_list):.4f}, R²: {np.mean(test_r2_list):.4f}, MAPE: {np.mean(test_mape_list):.2f}%")



