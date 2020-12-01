import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from numpy import apply_along_axis, linalg
from collections import Counter

df = pd.read_csv("normal_df.csv", sep = "\t", names = [0,1,2])

df_tmp = pd.read_csv("vitD_Sergio.csv", sep = "\t", names = [0,1,2,3])

target = df_tmp[3]

data = apply_along_axis(lambda x: x,1,df)

# la funcion de vecindario una funcion Gaussiana, sigma es a desviacion estardar
som = MiniSom(20, 20, 3, sigma = 1.0, learning_rate = 0.85)

# entrenamiento con 100 iteraciones
som.train_random(data, 2000) 


#print(som.distance_map.T)
outliers_percentage = 0.08

quantization_errors = np.linalg.norm(som.quantization(data) - data, axis=1)
error_treshold = np.percentile(quantization_errors, 100*(1-outliers_percentage)+5)
is_outlier = quantization_errors > error_treshold

print (Counter(is_outlier))

plt.hist(quantization_errors)
plt.axvline(error_treshold, color='k', linestyle='--')
plt.title('Histograma')
plt.xlabel('error')
plt.ylabel('frequencia')
plt.show()

outliers = []
for i in range(len(target)):
	if (is_outlier[i] == True):
		outliers.append(target[i])

print (outliers)



plt.figure(figsize=(10, 10))
plt.scatter(data[~is_outlier, 1], data[~is_outlier, 2],
            label='inlier')
plt.scatter(data[is_outlier, 1], data[is_outlier, 2],
            label='outlier')
plt.legend()
plt.show()
