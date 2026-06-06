import numpy as np
import nnfs
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd 

data = pd.read_csv('df_train.csv')
cols_to_drop = [
    'date', 'nice_view', 'perfect_condition', 'real_bathrooms', 
    'has_lavatory', 'single_floor', 'month', 'quartile_zone'
]
data = data.drop(columns=cols_to_drop)
xtrain = data.drop(['price'], axis=1)
ytrain = data[['price']]

ss = StandardScaler()
ohe = OneHotEncoder()
base_encoded = ohe.fit_transform(xtrain[['has_basement']]).toarray()
ren_encoded = ohe.fit_transform(xtrain[['renovated']]).toarray()
beds_scaled = ss.fit_transform(xtrain[['bedrooms']])
grade_scaled = ss.fit_transform(xtrain[['grade']])
living_scaled = ss.fit_transform(xtrain[['living_in_m2']])

xtrain = np.hstack([base_encoded, ren_encoded, beds_scaled, grade_scaled, living_scaled])
ytrain = ss.fit_transform(ytrain)
print(f"X Matrix Shape (Must be 2D): {xtrain.shape}")
#print(f"Y Matrix Shape (Must be 2D): {y_train_clean.shape}")



network = nnfs.neural_network()
network.add_layer(7, 4, 'relu')
network.add_layer(4, 8, 'relu')
network.add_layer(8, 16, 'relu')
network.add_layer(16, 1, 'linear')

pred, loss = network.train(xtrain, ytrain, batch_size=13603, alpha=0.001, epoch=500)
loss = np.array(loss)
print(f"Actual Value: {ytrain} \n")
print(f"Predictted values: {pred} \n")
print(f"Loss: {loss} \n")      
print(f"Shape of loss: {loss.shape} \n")
plt.plot(loss)
plt.show()

#network.network_info()
