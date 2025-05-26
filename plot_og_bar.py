import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example data
data = pd.DataFrame({
    'Dataset': ['Enron', 'LastFM', 'Reddit', 'UCI', 'tgbl-wiki'] * 2,
    'Time': [1.442472665, 27.07388986, 18.08228814, 1.638175269, 4.864469989, 0.5302, 4.8937,
              3.1084, 0.2002, 2.0578],
    'Method': ['UTG'] * 5 + ['OpenDG'] * 5
})

# Create grouped bar plot
sns.barplot(data=data, x='Dataset', y='Time', hue='Method')
plt.ylabel('Time', fontsize=20)
plt.xlabel('Dataset', fontsize=20)
plt.yscale('log')
plt.legend()
plt.savefig('discretizationc_bar.pdf',bbox_inches='tight')
