import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from collections import Counter
import seaborn as sns
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("vitD_Sergio.csv", sep = "\t", names = [0,1,2,3])
target = df[3]
df = df.drop(3, axis = 1)

# Normalizamos el dataset
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)
normal_df = pd.DataFrame(scaled_df, columns = [0,1,2])





isofor = IsolationForest(n_estimators= 100, contamination = 0.08).fit(normal_df)

out_bool = isofor.predict(normal_df)


outliers = []

out_color = []
for i in range(len(target)):
	if (out_bool[i] == -1):
		out_color.append('red')
	else:
		out_color.append('darkgrey')

for i in range(len(target)):
	if (out_bool[i] == -1):
		outliers.append(target[i])

print(Counter(out_color))

print(outliers)

outli = pd.DataFrame(out_color)

df2 = pd.concat([normal_df,outli], axis = 1, ignore_index=True)


df2 = pd.DataFrame(df2, columns =[0,1,2,3])
df3 = df2.rename(columns = {0: 'pri', 1: 'seg', 2: 'terc', 3: 'out'})

# sns.pairplot(data = df3, vars=['pri', 'seg', 'terc'], hue = 'out', diag_kind = 'hist')
# plt.show()

fig, axes = plt.subplots(1, 3)

axes[0].scatter(df3['pri'], df3['seg'], c = df3['out'])
axes[0].set_title("VDR8 vs VDR24")

for i in range(len(out_color)):
	if (out_color[i] == 'red'):
   		axes[0].text(0.015 + df3['pri'][i], df3['seg'][i], '{}'.format(target[i]), va='center')


axes[1].scatter(df3['pri'], df3['terc'], c = df3['out'])
axes[1].set_title("VDR8 vs H3K4me324")

for i in range(len(out_color)):
	if (out_color[i] == 'red'):
   		axes[1].text(0.015 + df3['pri'][i], df3['terc'][i], '{}'.format(target[i]), va='center')

axes[2].scatter(df3['seg'], df3['terc'], c = df3['out'])
axes[2].set_title("VDR24 vs H3K4me324")

for i in range(len(out_color)):
	if (out_color[i] == 'red'):
   		axes[2].text(0.015 + df3['seg'][i], df3['terc'][i], '{}'.format(target[i]), va='center')

plt.suptitle("Detección de Anomalías por Isolation Forest")
plt.savefig("DA_IsolationForest.png", bbox_inches='tight')

plt.show()