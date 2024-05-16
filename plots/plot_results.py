import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", font_scale=1)
plt.rcParams["font.family"] = "serif"


methods = ["TGN", "DyGFormer", "NAT", "GraphMixer", "HTGN", "GCLSTM", "EGCN", "GCN"]

# UCI_test_MRRs = [0.091, 0.334, 0.356, 0.105, 0.093, 0.093, 0.121, 0.068]
# UCI_inference_time = [1.07, 155.58, 3.82, 32.88, 0.61, 0.35, 0.43, 0.50]
# EdgeBank = 0.165

# dfs = pd.DataFrame(data={'method': methods, 
#                          'MRR': UCI_test_MRRs, 
#                          'time': UCI_inference_time,})



enron_test_MRRs = [0.191, 0.331, 0.276, 0.296, 0.267, 0.170, 0.233, 0.164]
enron_inference_time = [1.71, 57.72, 8.39, 13.85, 0.87, 0.46, 0.45, 0.31 ]
EdgeBank = 0.157 

dfs = pd.DataFrame(data={'method': methods, 
                         'MRR': enron_test_MRRs, 
                         'time': enron_inference_time,})



fig, axes = plt.subplots(1, 2, figsize=(18,5))
axes[0].set_title('Test MRR')


bar1 = sns.barplot(ax=axes[0], 
            x = 'method',
            y = 'MRR', data=dfs, hue="method", width=0.8)
#palette=colors_from_values(np.asarray(UCI_test_MRRs)
# axes[0].axhline(y=UCI_EdgeBank, color='r', linestyle='--')
# axes[0].text(3,UCI_EdgeBank,'$EdgeBank_{tw}$')

axes[0].axhline(y=EdgeBank, color='r', linestyle='--')
axes[0].text(3,EdgeBank,'$EdgeBank_{tw}$')

for i in axes[0].containers:
    axes[0].bar_label(i,)



bar2 = sns.barplot(ax=axes[1], 
            x = 'method',
            y = 'time', data=dfs, hue="method")
axes[1].set_title('Test Inference Time')
for i in axes[1].containers:
    axes[1].bar_label(i,)

# plt.legend()
# plt.savefig("UCI.pdf")
plt.savefig("enron.pdf")
plt.close()





# sns.barplot(x = methods,
#             y = UCI_test_MRRs, palette=colors_from_values(np.asarray(UCI_test_MRRs), "flare"))

# plt.axhline(y=UCI_EdgeBank, color='r', linestyle='--')
# plt.ylabel("MRR")
# plt.xlabel("Method")
# plt.title("UCI Test MRRs")
# plt.savefig("UCI_test_MRRs.png")
