import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter


def mad_based_outlier(points, thresh=3.5):
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum(abs(points - median), axis=-1)
	med_abs_deviation = np.median(diff)
	modified_z_score = 0.6745 * diff / med_abs_deviation
	for i in range(len(modified_z_score)):
		modified_z_score[i] = round(modified_z_score[i], 6)


	return [modified_z_score > thresh, median, modified_z_score]



df = pd.read_csv("vitD_Sergio.csv", sep = "\t", names =[0,1,2,3])


tmp_df = df.drop([3], axis = 1)

target = df[3]

# Normalizamos el dataset
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(tmp_df)
data = pd.DataFrame(scaled_df, columns = [0,1,2])
for x in list(data):
	for y in range(len(data[x])):
		data[x][y] = round(data[x][y], 6)

fig, ax = plt.subplots(1,2)

for col in list(data):
	outliers_bool, mad, modified_z_score = mad_based_outlier(data[col])
	out_color = []
	for i in range(len(target)):
		if (outliers_bool[i] == True):
			out_color.append('red')
		else:
			out_color.append('darkgrey')
	i = 0
	outliers = []
	if(col == 0 or col == 1):
		for out in outliers_bool:
			if(out == True):
				outliers.append(target[i])
			i += 1
		print(Counter(outliers_bool))
		print(outliers)

		ax[col].scatter(range(len(data[col])), data[col], c = out_color)
		for i in range(len(out_color)):
			if(out_color[i] == 'red'):
				ax[col].text(2 + i, data[col][i], '{}'.format(target[i]), va='center')
				if (col == 0 ):
					nom_col = 'VDR8'
				elif(col == 1):
					nom_col = 'VDR24'
				ax[col].set_title("Anomalías en {}".format(nom_col))
	
	

plt.suptitle("Detección de Anomalías por MAD (El atributo H3K4me324 no cuenta con anomalías)")
plt.savefig("DA_MAD.png", bbox_inches='tight')

plt.show()	

