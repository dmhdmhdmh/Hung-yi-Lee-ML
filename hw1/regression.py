import sys
import pandas as pd
import numpy as np
import math
import csv

data = pd.read_csv('D:/homeworkpy/Data/hw1/train.csv',encoding='big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
#raw_data = data.to_numpy()
raw_data=data.values.tolist()
raw_data=np.array(raw_data)
print(raw_data)

month_data = {}
for month in range(12):
    sample = np.empty([18,480])
    for day in range(20):
        sample[:,day * 24 : ( day + 1 ) * 24] = raw_data [ 18 * ( 20 * month + day ) : 18 * ( 20 * month + day + 1 ),: ]
    month_data[month] = sample

x = np.empty([12*471,18*9],dtype = float)
y = np.empty([12*471,1],dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour>14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9,day * 24 + hour + 9]
print(x)
print(y)

mean_x = np.mean(x,axis = 0)
std_x = np.std(x,axis = 0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
#将训练集分成训练-验证集，用来最后检验我们的模型
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))

#因为存在偏差bias，所以dim+1
dim = 18 * 9 + 1
# w维度为163*1
w = np.zeros([dim,1])
# x_train_set维度为 4521*163
x_train_set= np.concatenate((np.ones([len(x_train_set),1]),x_train_set),axis = 1).astype(float)
#设置学习率
learning_rate = 10
#设置迭代数
iter_time = 30000
#RMSprop参数初始化
adagrad = np.zeros([dim,1])
eps = 0.0000000001
#beta = 0.9
#迭代
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set,w)-y_train_set,2))/len(x_train_set))
    if(t%100 == 0):
        print("迭代的次数：%i ， 损失值：%f"%(t,loss))
        #gradient = 2*np.dot(x.transpose(),np.dot(x,w)-y)
        #计算梯度值
        gradient = (np.dot(x_train_set.transpose(),np.dot(x_train_set,w)-y_train_set))/(loss*len(x_train_set))
        adagrad += (gradient ** 2)
        #更新参数w
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
#保存参数w
np.save('weight.npy',w)

testdata = pd.read_csv('D:/homeworkpy/Data/hw1/test.csv',header = None ,encoding = 'big5')
test_data = testdata.iloc[:,2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.values.tolist()
test_data = np.array(test_data)
test_x = np.empty([240,18*9],dtype = float)
for i in range(240):
    test_x[i,:] = test_data[18*i:18*(i+1),:].reshape(1,-1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240,1]),test_x),axis = 1).astype(float)
print(test_x)

#在验证集上进行验证
w = np.load('weight.npy')
x_validation= np.concatenate((np.ones([len(x_validation),1]),x_validation),axis = 1).astype(float)
for m in range(len(x_validation)):
    Loss = np.sqrt(np.sum(np.power(np.dot(x_validation,w)-y_validation,2))/len(x_validation))
print ("the Loss on val data is %f" % (Loss))
#预测
ans_y = np.dot(test_x, w)
print('预测PM2.5值')
print(ans_y)


with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

