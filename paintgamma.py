import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gamma = pd.read_csv("./prediction/excel/gamma_506/gamma.csv").round(decimals=4)
gamma.head()

# citeseer = gamma[(gamma['dataset'] == 'citeseer')]
# cora = gamma[(gamma['dataset'] == 'cora')]

#  citeseer
datasets = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor']
for dataset in datasets:

    dataall = gamma[(gamma['dataset'] == dataset)]
    times = dataall['times'].unique()
    for time in times:
        plt.figure(figsize=(5, 4))
        data = dataall[(dataall['times'] == time)]
        xtemp = np.sort(data['gamma'].unique())
        ytemp = []
        for i in xtemp:
            acc_mean = data[(data['gamma'] == i)]['acc_mean']
            acc = acc_mean.to_list()
            ytemp.append(max(acc))

        xtemp = list(reversed(xtemp*(-1)))
        ytemp = list(reversed(ytemp))
        plt.plot(xtemp, ytemp, marker='p',linestyle='dashed', markersize=6)
        plt.xlabel('Gamma',fontsize='large')  # x轴标题
        #plt.xlabel('{}_{}'.format(dataset, time), fontsize='large')  # x轴标题
        plt.ylabel('Accuracy',fontsize='large')  # y轴标题
        plt.grid()
        plt.tight_layout()
        plt.savefig('./figures/parameter/gamma/gamma_{}.svg'.format(dataset),format='svg')
        plt.show()
        plt.clf()

# #  cora
# plt.figure(figsize=(5,4))
# data = cora
# xtemp = np.sort(data['gamma'].unique())
# ytemp = []
# for i in xtemp:
#     acc_mean = data[(data['gamma'] == i)]['acc_mean']
#     acc = acc_mean.to_list()
#     ytemp.append(max(acc))
#
# xtemp = list(reversed(xtemp*(-1)))
# ytemp = list(reversed(ytemp))
# plt.plot(xtemp, ytemp, marker='p',linestyle='dashed', markersize=6)
# plt.xlabel('Gamma',fontsize='large')  # x轴标题
# plt.ylabel('Accuracy',fontsize='large')  # y轴标题
# plt.grid()
# plt.tight_layout()
# plt.savefig('./gamma/cora.svg',format='svg')