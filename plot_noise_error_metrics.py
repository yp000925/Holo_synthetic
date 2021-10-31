#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
#%%
recall = [0.9172,0.9214,0.9201,0.9155,0.9175,0.9163]

precision = [1,1,1,1,1,0.9989]

x = np.array([5,10,15,20,25,30])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0, 1.1)
ax.plot(x,recall,'-o',c='green',label ='recall')
ax.plot(x,precision,'-o',c='blue',label ='precision')
#%%
errors = pd.read_csv("/Users/zhangyunping/PycharmProjects/Holo_synthetic/error_db/noise_error.csv")
errors = errors[['5db', '10db', '15db','20db', '25db', '30db']]
std = errors.describe().loc['std'].values/256*2*10
mean = errors.describe().loc['mean'].values/256*2*10
dy = std
ax2 = ax.twinx()
ax2.errorbar(x, mean, dy, fmt='o', ecolor='k', color='k', elinewidth=2, capsize=4,label ='depth error',ls='-',lw=2)
plt.show()
ax2.set_ylim(0.2, 2.1)
ax.legend(loc='lower right',fontsize=12)
ax2.legend(loc='center',fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(),fontsize=16)
ax2.set_yticklabels(ax.get_yticklabels(),fontsize=16)