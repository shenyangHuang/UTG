import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", font_scale=1.1)
rotation = 30

plt.rcParams["font.family"] = "serif"
label_size = 20
y_tick_size = 12

granularities = [1,2,4,6,12,24,48,96]

test_mrr = [0.464, 0.416, 0.423, 0.400, 0.406, 0.388, 0.321, 0.285]


dfs = pd.DataFrame(data={'Time Granularity': granularities, 
                         'Test MRR': test_mrr})

bar1 = sns.barplot(
            x = 'Time Granularity',
            y = 'Test MRR', data=dfs)

# bar1.set_yscale("log")
bar1.set_xlabel("Time Granularity (hours)",fontsize=label_size)
bar1.set_ylabel("Test MRR",fontsize=label_size)
plt.xticks(rotation=rotation)
plt.yticks(fontsize=y_tick_size)
plt.tight_layout()

for i in bar1.containers:
    bar1.bar_label(i,)

plt.savefig("time_granularity_MRR.pdf")
plt.close()