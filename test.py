import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

# 解决可能的库冲突问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 读取数据（注意：请确保文件路径正确）
features = pd.read_csv('data/data2.csv')

# 处理时间数据
import datetime

# 分别得到年，月，日
years = features['year']
months = features['month']
days = features['day']

# 转换为datetime格式
dates = [
    str(int(year)) + '-' + str(int(month)) + '-' + str(int(day))
    for year, month, day in zip(years, months, days)
]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 准备画图
plt.style.use('fivethirtyeight')

# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)

# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel('')
ax1.set_ylabel('Temperature')
ax1.set_title('Max Temp')

# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('Previous Max Temp')

# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('Two Days Prior Max Temp')

plt.tight_layout(pad=2)

# 独热编码（字符串数据转为数值形式）
features = pd.get_dummies(features)

# 标签
labels = np.array(features['actual'])

# 在特征中去掉标签
features = features.drop('actual', axis=1)

# 保存特征列名
feature_list = list(features.columns)

# 转换为数组
features = np.array(features)

from sklearn import preprocessing

# 标准化特征
input_features = preprocessing.StandardScaler().fit_transform(features)

# 定义神经网络参数
input_size = input_features.shape[1]  # 特征数量，不是样本个数
hidden_size = 128
output_size = 1
batch_size = 16

# 定义神经网络
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)

# 损失函数和优化器
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# 确定设备（CPU或GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将模型移动到设备
my_nn.to(device)

# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # 使用MINI-Batch方法训练
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)

        # 将数据转换为张量并移动到设备
        xx = torch.tensor(input_features[start:end], dtype=torch.float).to(device)
        yy = torch.tensor(labels[start:end], dtype=torch.float).to(device)

        # 前向传播
        prediction = my_nn(xx)
        loss = cost(prediction.squeeze(), yy)  # 确保维度匹配

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        batch_loss.append(loss.data.cpu().numpy())

    # 打印损失
    if i % 100 == 0:
        avg_loss = np.mean(batch_loss)
        losses.append(avg_loss)
        print(f"迭代次数: {i}, 平均损失: {avg_loss}")

# 生成预测
with torch.no_grad():  # 关闭梯度计算，节省内存
    x = torch.tensor(input_features, dtype=torch.float).to(device)
    predict = my_nn(x).cpu().data.numpy()  # 将结果移回CPU并转换为numpy

# 准备可视化数据
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 处理预测数据的日期
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [
    str(int(year)) + '-' + str(int(month)) + '-' + str(int(day))
    for year, month, day in zip(years, months, days)
]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={
    'date': test_dates,
    'prediction': predict.reshape(-1)
})

# 绘制结果
plt.figure(figsize=(20, 5))

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation=60)
plt.legend()

# 图名和标签
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()
