import numpy as np
import pandas as pd
from pyod.models import feature_bagging
from numpy import apply_along_axis, linalg
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("normal_df.csv", sep = "\t", names = [0,1,2])

df_tmp = pd.read_csv("vitD_Sergio.csv", sep = "\t", names = [0,1,2,3])
	
target = df_tmp[3]


fb = feature_bagging.FeatureBagging(contamination = 0.08)
fb.fit(df)

out_sino = fb.labels_

out_color = []

# for i in range(len(out_sino)):
# 	if(out_sino[i] == True):
# 		outliers.append(target[i])

outliers = []

for i in range(len(target)):
	if (out_sino[i] == True):
		out_color.append('red')
		outliers.append(target[i])
	else:
		out_color.append('darkgrey')


outli = pd.DataFrame(out_color)

df_grap = pd.concat([df, outli], axis = 1, ignore_index = True)

df_grap = df_grap.rename(columns = {0: 'pri', 1: 'seg', 2: 'terc', 3: 'out'})

fig, axes = plt.subplots(1, 3)

axes[0].scatter(df_grap['pri'], df_grap['seg'], c = df_grap['out'])
axes[0].set_title("VDR8 vs VDR24")

for i in range(len(out_color)):
	if (out_color[i] == 'red'):
   		axes[0].text(0.015 + df_grap['pri'][i], df_grap['seg'][i], '{}'.format(target[i]), va='center')


axes[1].scatter(df_grap['pri'], df_grap['terc'], c = df_grap['out'])
axes[1].set_title("VDR8 vs H3K4me324")

for i in range(len(out_color)):
	if (out_color[i] == 'red'):
   		axes[1].text(0.015 + df_grap['pri'][i], df_grap['terc'][i], '{}'.format(target[i]), va='center')

axes[2].scatter(df_grap['seg'], df_grap['terc'], c = df_grap['out'])
axes[2].set_title("VDR24 vs H3K4me324")

for i in range(len(out_color)):
	if (out_color[i] == 'red'):
   		axes[2].text(0.015 + df_grap['seg'][i], df_grap['terc'][i], '{}'.format(target[i]), va='center')


plt.suptitle("Detección de Anomalías por Feature Bagging con Estimador LOF")
plt.savefig("DA_FeatureBagging.png", bbox_inches='tight')


plt.show()

print (df_grap[df_grap['out'] == 'red'])
print(outliers)
#sns.pairplot(data = df_grap, vars=['pri', 'seg', 'terc'], hue = 'out', diag_kind = 'hist')
#plt.show()
#print(df_grap)