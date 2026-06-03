import numpy as np 

class layerclass:  #the class layer contains methods that the neural_network class requires to successfully produce an output
   
    #n_input is the number of inputs the neuron receives
    # n_output shows how many neurons are there in the layer

    def __init__(self, n_input, n_output):
        scale = np.sqrt(1/n_input)
        self.weight = np.random.rand(n_input, n_output) * scale 
        self.bias = np.random.rand(1, n_output)

    #input_value is the input as a matrix passed on to the neural network from the dataset
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a 
    
    def forward_pass(self, input_value):
        self.input_value = input_value  
        z = np.dot(input_value, self.weight) + self.bias
        self.a = self.sigmoid(z)
        return self.a
    
    def loss_function(self, actual_output, predicted_output):
        loss = np.mean((predicted_output - actual_output)**2) 
        return loss
    
    def back_prop(self,error_gradient,learning_rate = 0.1):
        org_weight = self.weight.copy()
        delta = error_gradient * self.a*(1 - self.a) 
        dw = np.dot(self.input_value.T, delta)
        db = db = np.sum(delta, axis = 0, keepdims=True)
        self.weight = self.weight - (learning_rate * dw)
        self.bias = self.bias -(learning_rate * db)
        error_gradient_to_pass = np.dot(delta, org_weight.T)
        return error_gradient_to_pass



class neural_network:
    def __init__(self):
        self.layer_list = []
    
    def add_layer(self, inp, out):  
        new_layer  = layerclass(inp, out)
        self.layer_list.append(new_layer)

    def train(self, xtrain, ytrain,batch_size, alpha, epoch = 100): #todo batch size 
        losses = []
        for i in range(1, epoch + 1):
            print(f"Epoch: {i}")
            # foreward propogation
            current_data = xtrain
            for lay in self.layer_list:
                current_data = lay.forward_pass(current_data)
            calculated_loss = lay.loss_function(current_data, ytrain)
            error_gradient = (current_data - ytrain) * (2/ batch_size)  
            for layers in self.layer_list[::-1]:
                new_error_gradient = layers.back_prop(error_gradient, alpha)
                error_gradient = new_error_gradient
            losses.append(calculated_loss)
        return current_data, losses