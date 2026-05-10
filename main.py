
import numpy as np
def network_initiate(no_input, no_of_layers, neurons, output):
        
    network = {}
    
    no_of_previous_nodes = no_input
    
    for layer in range(no_of_layers + 1): 
        if layer == (no_of_layers):
            layer_name = 'output_layer'
            num_of_nodes = output
        else: 
            layer_name = f'layer{layer + 1}'
            num_of_nodes = neurons[layer]
            
        network[layer_name] = {}
        
        for node in range(num_of_nodes): 
            #initializing weights and bias for each neuron 
            node_name = f"node{node + 1}"
            network[layer_name][node_name] = {
                'weight' : np.random.uniform(size = no_of_previous_nodes),
                'bias' : np.random.uniform(size = 1)
            }
            no_of_previous_nodes = neurons #Each neuron needs one weight per neuron in the previous layer. So the next layer needs to know how many neurons the current layer has.
    
    
    return network

def weighted_sum(inputs, weight, bias):
    return np.sum(weight * inputs) + bias

def activation(weighted_sum):
    return 1/(1 + np.exp(-weighted_sum))
def input_gen(no_input):
    return np.random.uniform(size = no_input)

def foreward_prop(network, inputs):
    inp = list(inputs)
    #output = []
    for layer in network:
        layer_data = network[layer]
        output = []
        
        for node in layer_data:
            node_data = layer_data[node]
            z = weighted_sum(inputs, weight = node_data['weight'], bias = node_data['bias'])
            neuron_output = activation(z)
            output.append(neuron_output)

    return output

#making prediction using the network
my_net = network_initiate(2, 3, [2, 2, 2], 1)
inp = input_gen(2)
x = foreward_prop(my_net, inp)
print(x)
