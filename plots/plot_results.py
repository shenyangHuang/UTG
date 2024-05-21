import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", font_scale=0.7)

plt.rcParams["font.family"] = "serif"
label_size = 20
y_tick_size = 12

methods = ["TGN", "DyGFormer", "NAT", "GraphMixer", "HTGN", "GCLSTM", "EGCN", "GCN", "EdgeBank"]

# UCI_test_MRRs = [0.091, 0.334, 0.356, 0.105, 0.093, 0.093, 0.121, 0.068, ]
# EdgeBank = 0.165
r"""
generate run time plots for DTDG
"""

UCI_inference_time = [1.07, 155.58, 3.82, 32.88, 0.61, 0.35, 0.43, 0.50, 0.52]
enron_inference_time = [1.71, 57.72,8.39, 13.85, 0.87,0.46, 0.45, 0.31, 0.25]
social_evo_inference_time = [24.04, 349.22, 148.43, 132.39, 14.59, 9.27, 7.35, 6.40,2.45]

UCI_inference_time = np.asarray(UCI_inference_time)
enron_inference_time = np.asarray(enron_inference_time)
social_evo_inference_time = np.asarray(social_evo_inference_time)

# avg_dtdg_time = np.mean([UCI_inference_time, enron_inference_time, social_evo_inference_time], axis=0)
avg_dtdg_time = social_evo_inference_time
avg_dtdg_time = np.round(avg_dtdg_time, decimals=1)


dfs = pd.DataFrame(data={'Method': methods, 
                         'Inference Time': avg_dtdg_time,})

# bar1 = sns.barplot(
#             x = 'Method',
#             y = 'Inference Time', data=dfs, hue="Method")
bar1 = sns.barplot(
            x = 'Method',
            y = 'Inference Time', data=dfs, hue='Inference Time')
bar1.set_yscale("log")
bar1.set_xlabel("Method",fontsize=label_size)
bar1.set_ylabel("Test Time",fontsize=label_size)
plt.legend(fontsize='large', title_fontsize='large')
plt.yticks(fontsize=y_tick_size)



# for i in bar1.containers:
#     bar1.bar_label(i,)

# plt.savefig("DTDG_time.pdf")
plt.savefig("social_evo_time.pdf")
plt.close()


r"""
generate run time plots for CTDG
"""
#! DyGFormer time is not reported yet, for now copies NAT time
wiki_time = [39.24, 7196.52, 340.51, 1655.44, 28.96,20.54,20.15,18.25,20.67]
review_time = [1137.69,26477.51,8925.21,4167.63,718.17,436.30,433.23,384.51,143.49]

wiki_time = np.asarray(wiki_time)
review_time = np.asarray(review_time)

# avg_ctdg_time = np.mean([wiki_time, review_time], axis=0)
avg_ctdg_time = review_time
avg_ctdg_time = np.round(avg_ctdg_time, decimals=1)


dfs = pd.DataFrame(data={'Method': methods, 
                         'Inference Time': avg_ctdg_time,})

# bar1 = sns.barplot(
#             x = 'Method',
#             y = 'Inference Time', data=dfs, hue="Method")
bar1 = sns.barplot(
            x = 'Method',
            y = 'Inference Time', data=dfs, hue='Inference Time')
bar1.set_yscale("log")
bar1.set_xlabel("Method",fontsize=label_size)
bar1.set_ylabel("Test Time",fontsize=label_size)
plt.yticks(fontsize=y_tick_size)


# for i in bar1.containers:
#     bar1.bar_label(i,)

# plt.savefig("CTDG_time.pdf")
plt.legend(fontsize='large', title_fontsize='large')
plt.savefig("review_time.pdf")
plt.close()



# dfs = pd.DataFrame(data={'Method': methods, 
#                          'MRR': UCI_test_MRRs, 
#                          'Inference Time': UCI_inference_time,})



# enron_test_MRRs = [0.191, 0.331, 0.276, 0.296, 0.267, 0.170, 0.233, 0.164]
# enron_inference_time = [1.71, 57.72, 8.39, 13.85, 0.87, 0.46, 0.45, 0.31 ]
# EdgeBank = 0.157 

# dfs = pd.DataFrame(data={'method': methods, 
#                          'MRR': enron_test_MRRs, 
#                          'time': enron_inference_time,})



# social_evo_test_MRRs = [0.283,0.366, 0.258, 0.157, 0.228, 0.286, 0.253, 0.289]
# social_evo_inference_time = [ 24.04, 349.22, 148.43, 132.39, 14.59, 9.27, 7.35, 6.40]
# EdgeBank = 0.070

# dfs = pd.DataFrame(data={'method': methods, 
#                          'MRR': social_evo_test_MRRs, 
#                          'time': social_evo_inference_time,})


# fig, axes = plt.subplots(1, 2, figsize=(20,5))
# axes[0].set_title('Test MRR')


# bar1 = sns.barplot(ax=axes[0], 
#             x = 'method',
#             y = 'MRR', data=dfs, hue="method", width=0.8)
# #palette=colors_from_values(np.asarray(UCI_test_MRRs)
# # axes[0].axhline(y=UCI_EdgeBank, color='r', linestyle='--')
# # axes[0].text(3,UCI_EdgeBank,'$EdgeBank_{tw}$')

# axes[0].axhline(y=EdgeBank, color='r', linestyle='--')
# axes[0].text(3,EdgeBank,'$EdgeBank_{tw}$')

# for i in axes[0].containers:
#     axes[0].bar_label(i,)



# bar2 = sns.barplot(ax=axes[1], 
#             x = 'method',
#             y = 'time', data=dfs, hue="method")
# axes[1].set_title('Test Inference Time')
# for i in axes[1].containers:
#     axes[1].bar_label(i,)

# # plt.legend()
# plt.savefig("UCI.pdf")
# # plt.savefig("enron.pdf")
# # plt.savefig("social_evo.pdf")
# plt.close()





# sns.barplot(x = methods,
#             y = UCI_test_MRRs, palette=colors_from_values(np.asarray(UCI_test_MRRs), "flare"))

# plt.axhline(y=UCI_EdgeBank, color='r', linestyle='--')
# plt.ylabel("MRR")
# plt.xlabel("Method")
# plt.title("UCI Test MRRs")
# plt.savefig("UCI_test_MRRs.png")
