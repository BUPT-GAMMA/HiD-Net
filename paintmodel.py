import matplotlib.pyplot as plt
import pandas as pd
# 按文件名读取整个文件
attack = pd.read_csv("prediction/excel/attack_4222/attack_4222_result.csv").round(decimals=4)
attack.head()
# 对mode进行分组
modeList = []
for mode in [0,1,2]:
    df = attack[(attack['mode'] == mode)]
    modeList.append(df)
datasetList = []
datasetname = ['cora','citeseer','squirrel']

for i in [0,1,2]:
    mode = modeList[i]
    #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # fig.tight_layout()#调整整体空白
    # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    # dataset = datasetList[0]
    for dataset in datasetname:
        df = mode[(mode['dataset'] == dataset)]
        datasetList.append(df)
        fig = plt.figure(figsize=(5,4))

        # -------------------------- cora_attack_deletion--------------------------------------
        ax1=plt.subplot(111)

        ratioList = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
        modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
        modelList2 = ['ADC','APPNP','DGC','GCN','HiD-Net','GRAND']
        yList = []
        ax1.set_xticks(ratioList)
        for model in modelList:
            # df = datasetList[(datasetList['model'] == model)]
            dfmodel = df[(df['model'] == model)]
            temp = []
            for ratio in ratioList:
                acc_mean = dfmodel[(dfmodel['ratio'] == ratio)]['acc_mean']
                acc = acc_mean.to_list()
                temp.append(max(acc))
            yList.append(temp)

        #画图

        makerList = ['p','<','>','*','v','+','h']
        for y,m in zip(yList,makerList):
            ax1.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
        #plt.ylim(ymin=0.78,ymax = 0.85)
        if i == 0:
            ax1.set_xlabel('Addition Rate',fontsize='large')  # x轴标题
        elif i == 1:
            ax1.set_xlabel('Deletion Rate', fontsize='large')  # x轴标题
        elif i == 2:
            ax1.set_xlabel('Feature Noise Rate', fontsize='large')  # x轴标题
        # ax1.set_xlabel('{}_{}.svg'.format(dataset, i), fontsize='large')  # x轴标题
        ax1.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
        ax1.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
        ax1.grid()
        plt.subplots_adjust(left=0.15, )
        plt.savefig('./figures/attack_518/{}_{}.svg'.format(dataset, i),format='svg',dpi=600)
        plt.show()  # 显示折线图



    # # -------------------------- citeseer_attack_deletion--------------------------------------
    # fig = plt.figure(figsize=(5,4))
    # ax2=plt.subplot(111)
    # dataset = datasetList[1]
    # ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    # modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
    # yList = []
    # ax2.set_xticks(ratioList)
    # for model in modelList:
    #     df  = dataset[(dataset['model'] == model)]
    #     temp = []
    #     for ratio in ratioList:
    #         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
    #         acc = acc_mean.to_list()
    #         temp.append(max(acc))
    #     yList.append(temp)
    #
    # #画图
    # makerList = ['p','<','>','*','v','+','h']
    # for y,m in zip(yList,makerList):
    #     ax2.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
    # #plt.ylim(ymin=0.78,ymax = 0.85)
    # ax2.set_xlabel('Addition Rate',fontsize='large')  # x轴标题
    # ax2.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
    # ax2.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
    # ax2.grid()
    # plt.savefig('./figures/attack/citeseer_{}.svg'.format(i),format='svg',dpi=600)
    # plt.show()  # 显示折线图
    #
    # # -------------------------- squirrel_attack_deletion--------------------------------------
    # fig = plt.figure(figsize=(5,4))
    # ax3=plt.subplot(111)
    # dataset = datasetList[2]
    # ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    # modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
    # yList = []
    # ax3.set_xticks(ratioList)
    # for model in modelList:
    #     df  = dataset[(dataset['model'] == model)]
    #     temp = []
    #     for ratio in ratioList:
    #         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
    #         acc = acc_mean.to_list()
    #         temp.append(max(acc))
    #     yList.append(temp)
    #
    #
    # #画图
    # makerList = ['p','<','>','*','v','+','h']
    # for y,m in zip(yList,makerList):
    #     ax3.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
    # #plt.ylim(ymin=0.78,ymax = 0.85)
    # ax3.set_xlabel('Addition Rate',fontsize='large')  # x轴标题
    # ax3.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
    # ax3.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
    # ax3.grid()
    # # plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.85, hspace=0.35, wspace=0.3)
    # plt.savefig('./figures/attack/citeseer_{}.svg'.format(i),format='svg',dpi=600)
    # plt.show()
    #



#
#
# mode = modeList[1]
# datasetList = []
# for dataset in ['cora','citeseer','squirrel']:
#     df = mode[(mode['dataset'] == dataset)]
#     datasetList.append(df)
#
#
# #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
# # fig.tight_layout()#调整整体空白
# # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
#
#
# # -------------------------- cora_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax1=plt.subplot(131)
# dataset = datasetList[0]
# ratioList = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax1.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
#
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax1.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax1.set_xlabel('Deletion Rate',fontsize='large')  # x轴标题
# ax1.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax1.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax1.grid()
# # plt.savefig('./attack/cora_attack_feature.svg',format='svg',dpi=600)
# # plt.show()  # 显示折线图
#
#
# # -------------------------- citeseer_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax2=plt.subplot(132)
# dataset = datasetList[1]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax2.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax2.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# import pandas as pd
# # 按文件名读取整个文件
# attack = pd.read_csv("prediction/excel/attack_4222/attack_4222_result.csv").round(decimals=4)
# attack.head()
# # 对mode进行分组
# modeList = []
# for mode in [0,1,2]:
#     df = attack[(attack['mode'] == mode)]
#     modeList.append(df)
#
# mode = modeList[0]
# datasetList = []
# for dataset in ['cora','citeseer','squirrel']:
#     df = mode[(mode['dataset'] == dataset)]
#     datasetList.append(df)
#
#
# #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
# # fig.tight_layout()#调整整体空白
# # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
#
# fig = plt.figure(figsize=(16.2,4))
#
# # -------------------------- cora_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax1=plt.subplot(131)
# dataset = datasetList[0]
# ratioList = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# modelList2 = ['adc','appnp','dgc','gcn','ggdn','grand']
# yList = []
# ax1.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
#
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax1.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax1.set_xlabel('Addition Rate',fontsize='large')  # x轴标题
# ax1.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax1.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax1.grid()
# # plt.savefig('./attack/cora_attack_feature.svg',format='svg',dpi=600)
# # plt.show()  # 显示折线图
#
#
#
# # -------------------------- citeseer_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax2=plt.subplot(132)
# dataset = datasetList[1]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax2.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax2.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax2.set_xlabel('Addition Rate',fontsize='large')  # x轴标题
# ax2.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax2.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax2.grid()
# # ax2.savefig('./attack/citeseer_attack_feature.svg',format='svg',dpi=600)
# # ax2.show()  # 显示折线图
#
# # -------------------------- squirrel_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax3=plt.subplot(133)
# dataset = datasetList[2]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax3.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax3.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax3.set_xlabel('Addition Rate',fontsize='large')  # x轴标题
# ax3.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax3.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax3.grid()
# plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.85, hspace=0.35, wspace=0.3)
# plt.savefig('figures/attack/attack_addition.svg',format='svg',dpi=600)
# plt.show()
#
# mode = modeList[1]
# datasetList = []
# for dataset in ['cora','citeseer','squirrel']:
#     df = mode[(mode['dataset'] == dataset)]
#     datasetList.append(df)
#
#
# #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
# # fig.tight_layout()#调整整体空白
# # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
#
# fig = plt.figure(figsize=(16.2, 4))
#
# # -------------------------- cora_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax1=plt.subplot(131)
# dataset = datasetList[0]
# ratioList = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax1.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
#
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax1.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax1.set_xlabel('Deletion Rate',fontsize='large')  # x轴标题
# ax1.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax1.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax1.grid()
# # plt.savefig('./attack/cora_attack_feature.svg',format='svg',dpi=600)
# # plt.show()  # 显示折线图
#
#
# # -------------------------- citeseer_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax2=plt.subplot(132)
# dataset = datasetList[1]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax2.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax2.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax2.set_xlabel('Deletion Rate',fontsize='large')  # x轴标题
# ax2.set_ylabel('Accuracy',fontsize='large')  # y轴标题
# ax2.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax2.grid()
# # ax2.savefig('./attack/citeseer_attack_feature.svg',format='svg',dpi=600)
# # ax2.show()  # 显示折线图
#
# # -------------------------- squirrel_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax3=plt.subplot(133)
# dataset = datasetList[2]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax3.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax3.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax3.set_xlabel('Deletion Rate',fontsize='large')  # x轴标题
# ax3.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax3.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax3.grid()
# plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.85, hspace=0.35, wspace=0.3)
# plt.savefig('./figures/attack/attack_deletion.svg',format='svg',dpi=600)
# plt.show()
#
# mode = modeList[2]
# datasetList = []
# for dataset in ['cora','citeseer','squirrel']:
#     df = mode[(mode['dataset'] == dataset)]
#     datasetList.append(df)
#
#
# #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
# # fig.tight_layout()#调整整体空白
# # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
#
# fig = plt.figure(figsize=(16.2,4))
#
# # -------------------------- cora_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax1=plt.subplot(131)
# dataset = datasetList[0]
# ratioList = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax1.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
#
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax1.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax1.set_xlabel('Feature Noise Rate',fontsize='large')  # x轴标题
# ax1.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax1.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax1.grid()
# # plt.savefig('./attack/cora_attack_feature.svg',format='svg',dpi=600)
# # plt.show()  # 显示折线图
#
#
# # -------------------------- citeseer_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax2=plt.subplot(132)
# dataset = datasetList[1]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax2.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax2.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax2.set_xlabel('Feature Noise Rate',fontsize='large')  # x轴标题
# ax2.set_ylabel('Accuracy',fontsize='large')  # y轴标题
# ax2.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax2.grid()
# # ax2.savefig('./attack/citeseer_attack_feature.svg',format='svg',dpi=600)
# # ax2.show()  # 显示折线图
#
# # -------------------------- squirrel_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax3=plt.subplot(133)
# dataset = datasetList[2]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax3.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax3.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax3.set_xlabel('Feature Noise Rate',fontsize='large')  # x轴标题
# ax3.set_ylabel('Accuracy',fontsize='large')  # y轴标题
# ax3.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax3.grid()
# plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.85, hspace=0.35, wspace=0.3)
# plt.savefig('./figures/attack/attack_feature.svg',format='svg',dpi=600)
# plt.show()
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax2.set_xlabel('Deletion Rate',fontsize='large')  # x轴标题
# ax2.set_ylabel('Accuracy',fontsize='large')  # y轴标题
# ax2.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax2.grid()
# # ax2.savefig('./attack/citeseer_attack_feature.svg',format='svg',dpi=600)
# # ax2.show()  # 显示折线图
#
# # -------------------------- squirrel_attack_deletion--------------------------------------
# ax3=plt.subplot(133)
# dataset = datasetList[2]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax3.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax3.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax3.set_xlabel('Deletion Rate',fontsize='large')  # x轴标题
# ax3.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax3.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax3.grid()
# plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.85, hspace=0.35, wspace=0.3)
# plt.savefig('./figures/attack/attack_deletion.svg',format='svg',dpi=600)
# plt.show()
#
# mode = modeList[2]
# datasetList = []
# for dataset in ['cora','citeseer','squirrel']:
#     df = mode[(mode['dataset'] == dataset)]
#     datasetList.append(df)
#
#
# #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
# # fig.tight_layout()#调整整体空白
# # plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
#
# fig = plt.figure(figsize=(16.2,4))
#
# # -------------------------- cora_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax1=plt.subplot(131)
# dataset = datasetList[0]
# ratioList = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax1.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
#
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax1.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax1.set_xlabel('Feature Noise Rate',fontsize='large')  # x轴标题
# ax1.set_ylabel('Test Accuracy',fontsize='large')  # y轴标题
# ax1.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax1.grid()
# # plt.savefig('./attack/cora_attack_feature.svg',format='svg',dpi=600)
# # plt.show()  # 显示折线图
#
#
# # -------------------------- citeseer_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax2=plt.subplot(132)
# dataset = datasetList[1]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax2.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax2.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax2.set_xlabel('Feature Noise Rate',fontsize='large')  # x轴标题
# ax2.set_ylabel('Accuracy',fontsize='large')  # y轴标题
# ax2.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax2.grid()
# # ax2.savefig('./attack/citeseer_attack_feature.svg',format='svg',dpi=600)
# # ax2.show()  # 显示折线图
#
# # -------------------------- squirrel_attack_deletion--------------------------------------
# fig = plt.figure(figsize=(5,4))
# ax3=plt.subplot(133)
# dataset = datasetList[2]
# ratioList  = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
# modelList = ['adc','appnp','dgc','gcn','gdgc','grand']
# yList = []
# ax3.set_xticks(ratioList)
# for model in modelList:
#     df  = dataset[(dataset['model'] == model)]
#     temp = []
#     for ratio in ratioList:
#         acc_mean = df[(df['ratio'] == ratio)]['acc_mean']
#         acc = acc_mean.to_list()
#         temp.append(max(acc))
#     yList.append(temp)
#
#
# #画图
# makerList = ['p','<','>','*','v','+','h']
# for y,m in zip(yList,makerList):
#     ax3.plot(ratioList, y, marker=m,linestyle='dashed', markersize=6)
# #plt.ylim(ymin=0.78,ymax = 0.85)
# ax3.set_xlabel('Feature Noise Rate',fontsize='large')  # x轴标题
# ax3.set_ylabel('Accuracy',fontsize='large')  # y轴标题
# ax3.legend(modelList2,loc="upper center", borderaxespad=-3.6,ncol=3,framealpha=0)  # 设置折线名称
# ax3.grid()
# plt.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.85, hspace=0.35, wspace=0.3)
# plt.savefig('./figures/attack/attack_feature.svg',format='svg',dpi=600)
# plt.show()
