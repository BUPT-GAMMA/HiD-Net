import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 按文件名读取整个文件
parameter = pd.read_csv("prediction/excel/parameter_428/parameter_428_result.csv").round(decimals=4)
parameter.head()
parameter['gamma'] = -parameter['gamma']

datasetList = np.sort(parameter['dataset'].unique())
dataset = []
for index in datasetList:
    dataset.append(parameter[(parameter['dataset'] == index)])

alphaValue = [0.3,0.1,0.25,0.15,0.08,0.1]
betaValue = [0.7,0.9,0.75,0.85,0.92,0.9]
gammaValue = [-0.1,0.3,-0.3,-0.3,-0.3,-0.5]

alphaList = []
betaList = []
# gammaList = []
for i in range(len(dataset)):
    data = dataset[i]
    alpha = data['alpha'].mode()[0]
    beta = data['beta'].mode()[0]
    gamma = data['gamma'].mode()[0]
    alphaList.append(data[(data['beta'] == beta)&(data['gamma'] == gamma)])
    betaList.append(data[(data['alpha'] == alpha)&(data['gamma'] == gamma)])
    #gammaList.append(data[(data['alpha'] == alpha)&(data['beta'] == beta)])

fig = plt.figure(figsize=(5,4))

xList = []
yList = []
for i in range(6):
    data = alphaList[i]
    temp = np.sort(data['alpha'].unique())
    ytemp = []
    for i in temp:
        acc_mean = data[(data['alpha'] == i)]['acc_mean']
        acc = acc_mean.to_list()
        ytemp.append(max(acc))
    xList.append(temp)
    yList.append(ytemp)
ax1=plt.subplot(111)
makerList = ['p','<','>','*','v','+','h']
for x,y,m in zip(xList,yList,makerList):
    ax1.plot(x, y, marker=m,linestyle='dashed', markersize=6)
#plt.ylim(ymin=0.78,ymax = 0.85)
ax1.set_xlabel('Alpha',fontsize='large')  # x轴标题
ax1.set_ylabel('Accuracy',fontsize='large')  # y轴标题
ax1.legend(datasetList,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
ax1.grid()
plt.savefig('figures/parameter/Alpha.svg',format='svg',dpi=600)
plt.show()




fig = plt.figure(figsize=(5,4))

xList = []
yList = []
for i in range(6):
    data = betaList[i]
    temp = np.sort(data['beta'].unique())
    ytemp = []
    for i in temp:
        acc_mean = data[(data['beta'] == i)]['acc_mean']
        acc = acc_mean.to_list()
        ytemp.append(max(acc))
    xList.append(temp)
    yList.append(ytemp)
ax1=plt.subplot(111)
makerList = ['p','<','>','*','v','+','h']
for x,y,m in zip(xList,yList,makerList):
    ax1.plot(x, y, marker=m,linestyle='dashed', markersize=6)
#plt.ylim(ymin=0.78,ymax = 0.85)
ax1.set_xlabel('Beta',fontsize='large')  # x轴标题
ax1.set_ylabel('Accuracy',fontsize='large')  # y轴标题
ax1.legend(datasetList,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
ax1.grid()
plt.savefig('figures/parameter/beta.svg',format='svg',dpi=600)
plt.show()

# xList = []
# yList = []
# for i in range(6):
#     data = betaList[i]
#     temp = np.sort(data['beta'].unique())
#     ytemp = []
#     for i in temp:
#         acc_mean = data[(data['beta'] == i)]['acc_mean']
#         acc = acc_mean.to_list()
#         ytemp.append(max(acc))
#     xList.append(temp)
#     yList.append(ytemp)
# ax2=plt.subplot(122)
# makerList = ['p','<','>','*','v','+','h']
# for x,y,m in zip(xList,yList,makerList):
#     ax2.plot(x, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax2.set_xlabel('Beta',fontsize='large')  # x轴标题
# ax2.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax2.legend(datasetList,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax2.grid()

# xList = []
# yList = []
# for i in range(6):
#     data = gammaList[i]
#     temp = np.sort(data['gamma'].unique())
#     ytemp = []
#     for i in temp:
#         acc_mean = data[(data['gamma'] == i)]['acc_mean']
#         acc = acc_mean.to_list()
#         ytemp.append(max(acc))
#     xList.append(temp)
#     yList.append(ytemp)
# ax3=plt.subplot(133)
# makerList = ['p','<','>','*','v','+','h']
# for x,y,m in zip(xList,yList,makerList):
#     ax3.plot(x, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax3.set_xlabel('Gamma',fontsize='large')  # x轴标题

# ax3.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax3.legend(datasetList,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax3.grid()
# plt.subplots_adjust(left=0.125,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.2,
#                     hspace=0.35)
