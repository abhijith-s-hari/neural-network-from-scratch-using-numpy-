#from multipledispatch import dispatch
import numpy as np 

# declaring and initiating a dictionary, that will contain all the required activation functions and its derivattive 
activation_functions = {
       'relu' : {
           'foreward' : lambda z: np.maximum(0, z),
           'backward' : lambda z: (z > 0).astype(float)
       },

       'sigmoid' : {
           'foreward' : lambda z : 1 / (1 + np.exp(-z)),
           'backward' : lambda z : (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
       },
       'tanh' : {
           'foreward' : lambda z : np.tanh(z), 
           'backward' : lambda z : 1 - np.tanh(z)**2 
       }
   }



class layerclass:  #the class layer contains methods that the neural_network class requires to successfully produce an output

    #n_input is the number of inputs the neuron receives
    # n_output shows how many neurons are there in the layer

    def __init__(self, n_input, n_output,act_fun):
        self.output_neuron = n_output 
        scale = np.sqrt(1/n_input)
        self.weight = np.random.randn(n_input, n_output) * scale 
        self.bias = np.random.randn(1, n_output)
        self.act_fun = act_fun

    '''
    def relu(self, z):
        a = np.maximum(0, z)
        return a 
    
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a 
    ''' 

    def forward_pass(self, input_value):
        self.input_value = input_value  
        self.z = np.dot(input_value, self.weight) + self.bias
        self.a = activation_functions[self.act_fun]['foreward'](self.z)
        return self.a
    
    def loss_function(self, actual_output, predicted_output):
        loss = np.mean((predicted_output - actual_output)**2) 
        return loss
    
    def back_prop(self,error_gradient, learning_rate = 0.1):
        org_weight = self.weight.copy()
        delta = error_gradient * activation_functions[self.act_fun]['backward'](self.z) 
        dw = np.dot(self.input_value.T, delta)
        db = db = np.sum(delta, axis = 0, keepdims=True)
        self.weight = self.weight - (learning_rate * dw)
        self.bias = self.bias -(learning_rate * db)
        error_gradient_to_pass = np.dot(delta, org_weight.T)
        return error_gradient_to_pass

    #def accuracy --------> todo 


class neural_network:

    def __init__(self):
        self.layer_list = []
        self.activation = ''
    
    def add_layer(self, inp, out, actv_function):  
        new_layer  = layerclass(inp, out, actv_function)
        self.layer_list.append(new_layer)
       # self.activation = actv_function

    def train(self, xtrain, ytrain,batch_size, alpha = 0.01, epoch = 100): #todo batch size 
        act_fun = self.activation
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
                new_error_gradient = layers.back_prop(error_gradient = error_gradient, learning_rate = alpha)
                error_gradient = new_error_gradient
            losses.append(calculated_loss)

        return current_data, losses

    # function to print out the weights and biases of each layer, also shows the no of nurons in each layer 
    def network_info(self):
        i = 1
        for lay in self.layer_list:
            print(f"Layer: {i}, No. of neurons: {lay.output_neuron}  \n")
            print(f"Weights: {lay.weight} \n")
            print(f"Bias: {lay.bias} \n")
            i+=1
            print('=======================================================================================================================')
