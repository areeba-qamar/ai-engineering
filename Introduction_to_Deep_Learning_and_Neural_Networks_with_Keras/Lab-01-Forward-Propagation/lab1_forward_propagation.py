# ============================================================
# Lab 1 - Forward Propagation
# ============================================================
#
# Lab Flow:
#
# 1. Understand how a neural network makes predictions
# 2. Manually calculate weighted sums
# 3. Apply the sigmoid activation function
# 4. Calculate the output/prediction of a simple network
# 5. Generalize the network so it can have:
#       - Multiple inputs
#       - Multiple hidden layers
#       - Multiple nodes per layer
#       - Multiple output nodes
# 6. Create a reusable function to initialize a network
# 7. Create a function to calculate weighted sums
# 8. Create a function for node activation
# 9. Combine everything into forward propagation
# 10. Test the network with different architectures
#
# Main idea:
#
# Input
#   ↓
# Weighted Sum
#   ↓
# Activation Function
#   ↓
# Hidden Layer Output
#   ↓
# Next Layer
#   ↓
# Final Prediction
#
# Note:
# We are building this from scratch for learning purposes.
# In real projects, libraries such as Keras/TensorFlow
# handle these operations for us.
# ============================================================



# ------------------------------------------------------------
# 1. Import Required Libraries
# ------------------------------------------------------------

# NumPy is used for numerical calculations.
# We will use it for arrays, weights, biases, and mathematical operations.
import numpy as np


# ------------------------------------------------------------
# 2. Define Input Values
# ------------------------------------------------------------

# These are the input values given to our neuron.
# Think of them as the information/features entering the neuron.
inputs = np.array([2.0, 3.0])

print("Inputs:", inputs)


# ------------------------------------------------------------
# 3. Define Weights
# ------------------------------------------------------------

# Each input has a corresponding weight.
# A weight determines how strongly an input influences the neuron.
weights = np.array([0.5, 0.8])

print("Weights:", weights)


# ------------------------------------------------------------
# 4. Define Bias
# ------------------------------------------------------------

# Bias is an additional value added to the weighted sum.
# It gives the neuron more flexibility when learning patterns.
bias = 0.2

print("Bias:", bias)


# Calculate the weighted sum of the inputs
weighted_sum = np.dot(inputs, weights) + bias

print("Weighted sum:", weighted_sum)


# ------------------------------------------------------------
# 5. Apply Sigmoid Activation Function
# ------------------------------------------------------------

# The sigmoid function converts the weighted sum into a value
# between 0 and 1.
#
# Formula:
# sigmoid(x) = 1 / (1 + e^(-x))
#
# This helps a neuron produce an activated output.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Apply the sigmoid function to our weighted sum.
output = sigmoid(weighted_sum)

print("Neuron output:", output)


# ------------------------------------------------------------
# 6. Create a Layer with Multiple Neurons
# ------------------------------------------------------------

# Our layer will have 2 neurons.
# Each neuron receives the same input values,
# but has its own weights and bias.

# Weights for Neuron 1
weights_1 = np.array([0.5, 0.8])
bias_1 = 0.2

# Weights for Neuron 2
weights_2 = np.array([0.7, 0.3])
bias_2 = 0.1


# Calculate the weighted sum for Neuron 1.
weighted_sum_1 = np.dot(inputs, weights_1) + bias_1

# Apply sigmoid activation to Neuron 1.
output_1 = sigmoid(weighted_sum_1)


# Calculate the weighted sum for Neuron 2.
weighted_sum_2 = np.dot(inputs, weights_2) + bias_2

# Apply sigmoid activation to Neuron 2.
output_2 = sigmoid(weighted_sum_2)


# Store the outputs of both neurons.
layer_outputs = np.array([output_1, output_2])

print("Layer outputs:", layer_outputs)


# ------------------------------------------------------------
# 7. Create a Second Layer
# ------------------------------------------------------------

# The outputs from the first layer now become
# the inputs to the second layer.
second_layer_inputs = layer_outputs


# The second layer also has 2 neurons.
# Each neuron has its own weights and bias.

# Weights for Second Layer Neuron 1
weights_3 = np.array([0.4, 0.6])
bias_3 = 0.1

# Weights for Second Layer Neuron 2
weights_4 = np.array([0.2, 0.7])
bias_4 = 0.2


# Calculate the weighted sum for Second Layer Neuron 1.
weighted_sum_3 = np.dot(second_layer_inputs, weights_3) + bias_3

# Apply sigmoid activation.
output_3 = sigmoid(weighted_sum_3)


# Calculate the weighted sum for Second Layer Neuron 2.
weighted_sum_4 = np.dot(second_layer_inputs, weights_4) + bias_4

# Apply sigmoid activation.
output_4 = sigmoid(weighted_sum_4)


# Store the outputs of the second layer.
second_layer_outputs = np.array([output_3, output_4])

print("Second layer outputs:", second_layer_outputs)


# ------------------------------------------------------------
# 8. Create the Final Output Layer
# ------------------------------------------------------------

# The outputs from the second hidden layer
# become the inputs to our final output neuron.
output_layer_inputs = second_layer_outputs


# The final neuron has its own weights and bias.
output_weights = np.array([0.6, 0.5])
output_bias = 0.1


# Calculate the weighted sum for the final neuron.
final_weighted_sum = np.dot(output_layer_inputs, output_weights) + output_bias

# Apply sigmoid activation to get the final prediction.
final_prediction = sigmoid(final_weighted_sum)


print("Final prediction:", final_prediction)


# ------------------------------------------------------------
# 9. Create a Function for Neuron Activation
# ------------------------------------------------------------

# This function takes:
#   inputs  -> values coming from the previous layer
#   weights -> importance of each input
#   bias    -> additional value
#
# It calculates the weighted sum and then
# applies the sigmoid activation function.
def activate_neuron(inputs, weights, bias):
    
    # Calculate the weighted sum.
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Apply sigmoid activation.
    output = sigmoid(weighted_sum)
    
    # Return the neuron's output.
    return output


# Test the function using our original neuron.
test_output = activate_neuron(inputs, weights, bias)

print("Reusable neuron output:", test_output)


# ------------------------------------------------------------
# 10. Create a Function to Calculate a Whole Layer
# ------------------------------------------------------------

# This function takes:
#   inputs  -> outputs coming from the previous layer
#   weights -> weights of all neurons in the current layer
#   biases  -> bias value for each neuron
#
# It calculates the output of every neuron in the layer.
def calculate_layer(inputs, weights, biases):

    # Store the outputs of all neurons.
    layer_outputs = []

    # Go through each neuron in the layer.
    for neuron_weights, neuron_bias in zip(weights, biases):

        # Calculate this neuron's output
        # using our reusable activation function.
        neuron_output = activate_neuron(
            inputs,
            neuron_weights,
            neuron_bias
        )

        # Add the neuron's output to the layer outputs.
        layer_outputs.append(neuron_output)

    # Convert the list into a NumPy array.
    return np.array(layer_outputs)


# Weights for both neurons in the first layer.
layer_1_weights = np.array([
    [0.5, 0.8],
    [0.7, 0.3]
])

# Bias for each neuron.
layer_1_biases = np.array([0.2, 0.1])


# Calculate the complete first layer using our function.
layer_1_outputs = calculate_layer(
    inputs,
    layer_1_weights,
    layer_1_biases
)

print("Reusable layer outputs:", layer_1_outputs)


# ------------------------------------------------------------
# 11. Create the Forward Propagation Function
# ------------------------------------------------------------

# This function passes the input data through
# every layer of the neural network.
#
# Each layer receives the outputs of the previous layer
# and produces new outputs.
def forward_propagate(inputs, network):

    # Start with the original input values.
    layer_inputs = inputs

    # Process each layer in the network.
    for layer_weights, layer_biases in network:

        # Calculate the outputs of the current layer.
        layer_outputs = calculate_layer(
            layer_inputs,
            layer_weights,
            layer_biases
        )

        # The current layer's outputs become
        # the next layer's inputs.
        layer_inputs = layer_outputs

    # After all layers are processed,
    # return the final output/prediction.
    return layer_inputs

# ------------------------------------------------------------
# 12. Build the Complete Neural Network
# ------------------------------------------------------------

# Layer 1: 2 neurons
network_layer_1 = (
    np.array([
        [0.5, 0.8],
        [0.7, 0.3]
    ]),
    np.array([0.2, 0.1])
)

# Layer 2: 2 neurons
network_layer_2 = (
    np.array([
        [0.4, 0.6],
        [0.2, 0.7]
    ]),
    np.array([0.1, 0.2])
)

# Output layer: 1 neuron
network_output_layer = (
    np.array([
        [0.6, 0.5]
    ]),
    np.array([0.1])
)


# Store all layers in the network.
my_network = [
    network_layer_1,
    network_layer_2,
    network_output_layer
]


# Perform forward propagation through the entire network.
prediction = forward_propagate(inputs, my_network)

print("Network prediction:", prediction)

# ------------------------------------------------------------
# 13. Test a Different Network Architecture
# ------------------------------------------------------------

# This time, we create a smaller network:
#
# Input (2 values)
#       ↓
# Hidden Layer (3 neurons)
#       ↓
# Output Layer (1 neuron)
#
# This shows that our forward propagation function
# is not limited to one specific network structure.

# First layer with 3 neurons.
different_layer_1 = (
    np.array([
        [0.2, 0.4],
        [0.5, 0.1],
        [0.3, 0.7]
    ]),
    np.array([0.1, 0.2, 0.3])
)

# Output layer with 1 neuron.
different_output_layer = (
    np.array([
        [0.4, 0.6, 0.5]
    ]),
    np.array([0.2])
)


# Build the new network.
different_network = [
    different_layer_1,
    different_output_layer
]


# Perform forward propagation through the new network.
different_prediction = forward_propagate(
    inputs,
    different_network
)

print("Prediction from different network:", different_prediction)