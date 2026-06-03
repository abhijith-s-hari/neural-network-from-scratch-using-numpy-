import numpy as np
import nnfd
import matplotlib.pyplot as plt 
x_test = np.array([
    [0.5, 0.2], [0.1, 0.9], [0.3, 0.4], [0.8, 0.1], [0.2, 0.7],
    [0.6, 0.5], [0.9, 0.9], [0.0, 0.2], [0.4, 0.6], [0.7, 0.3],
    [0.1, 0.1], [0.5, 0.8], [0.8, 0.6], [0.2, 0.3], [0.6, 0.9],
    [0.3, 0.2], [0.7, 0.7], [0.4, 0.1], [0.9, 0.4], [0.0, 0.8]
])
y_test = np.random.randint(0, 2, size=(20, 1))

network = nnfd.neural_network()
network.add_layer(2, 4)
network.add_layer(4, 8)
network.add_layer(8, 16)
network.add_layer(16, 1)

pred, loss = network.train(x_test, y_test, batch_size=20, alpha=0.01, epoch=1000)
loss = np.array(loss)
print(f"Predictted values: {pred} loss: {loss}" )
print(f"Shape of loss: {loss.shape}")
plt.plot(loss)
plt.show()
