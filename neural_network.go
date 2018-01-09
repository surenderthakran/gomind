// Package gomind for a simple Multi Layer Perceptron (MLP) Feed Forward Artificial Neural Network library.
package gomind

import (
	"fmt"
	"math/rand"
	"time"
)

// NeuralNetwork describes a single hidden layer MLP feed forward neural network.
type NeuralNetwork struct {
	numberOfInputs int
	hiddenLayer    *layer
	outputLayer    *layer
}

type Activation int

const (
	SIGMOID Activation = iota
)

const (
	learningRate = 0.5
)

// NewNeuralNetwork function returns a new NeuralNetwork object.
func NewNeuralNetwork(numberOfInputs, numberOfHiddenNeurons, numberOfOutputs int) (*NeuralNetwork, error) {
	// setting timestamp as seed for random number generator.
	rand.Seed(time.Now().UnixNano())

	hiddenLayer, err := newLayer(numberOfHiddenNeurons, numberOfInputs)
	if err != nil {
		return nil, fmt.Errorf("error creating a hidden layer: %v", err)
	}

	outputLayer, err := newLayer(numberOfOutputs, numberOfHiddenNeurons)
	if err != nil {
		return nil, fmt.Errorf("error creating output layer: %v", err)
	}

	return &NeuralNetwork{
		numberOfInputs: numberOfInputs,
		hiddenLayer:    hiddenLayer,
		outputLayer:    outputLayer,
	}, nil
}

// CalculateOutput function returns the output array from the neural network for the given
// input array based on the current weights.
func (network *NeuralNetwork) CalculateOutput(input []float64) []float64 {
	hiddenOutput := network.hiddenLayer.calculateOutput(input)
	return network.outputLayer.calculateOutput(hiddenOutput)
}

// LastOutput function returns the array of last output computed by the network.
func (network *NeuralNetwork) LastOutput() []float64 {
	var output []float64
	for _, neuron := range network.outputLayer.neurons {
		output = append(output, neuron.output)
	}
	return output
}

// Train function trains the neural network using the given set of inputs and outputs.
func (network *NeuralNetwork) Train(trainingInput, trainingOutput []float64) {
	outputs := network.CalculateOutput(trainingInput)
	network.calculateNewOutputLayerWeights(outputs, trainingOutput)
	network.calculateNewHiddenLayerWeights()
	network.updateWeights()
}

// calculateNewOutputLayerWeights function calculates new weights from the
// hidden layer to the output layer and bias for the output layer neurons, after
// calculating how much each weight and bias affects the total error in the
// final output of the network. i.e. the partial differential of error with
// respect to the weight. ∂Error/∂Weight and the partial differential of error
// with respect to the bias. ∂Error/∂Bias.
//
// By applying the chain rule, https://en.wikipedia.org/wiki/Chain_rule
// ∂TotalError/∂OutputNeuronWeight = ∂TotalError/∂TotalNetInputToOutputNeuron * ∂TotalNetInputToOutputNeuron/∂OutputNeuronWeight
func (network *NeuralNetwork) calculateNewOutputLayerWeights(outputs, targetOutputs []float64) {
	for neuronIndex, neuron := range network.outputLayer.neurons {
		// Since a neuron has only one total net input and one output, we need to calculate
		// the partial derivative of error with respect to the total net input (∂TotalError/∂TotalNetInputToOutputNeuron) only once.
		//
		// The total error of the network is a sum of errors in all the output neurons.
		// ex: Total Error = error1 + erro2 + error3 + ...
		// But when calculating the partial derivative of the total error with respect to the total net input
		// of only one output neuron, we need to find partial derivative of only the corresponding neuron's error because
		// the errors due to other neurons would be constant for it and their derivative wouldn't matter.
		pdErrorWrtTotalNetInputOfOutputNeuron := neuron.calculatePdErrorWrtTotalNetInputOfOutputNeuron(targetOutputs[neuronIndex])

		for weightIndex, weight := range neuron.weights {
			// For each weight of the neuron we calculate the partial derivative of
			// total net input with respect to the weight i.e. ∂TotalNetInputToOutputNeuron/∂OutputNeuronWeight.
			pdTotalNetInputWrtWeight := neuron.calculatePdTotalNetInputWrtWeight(weightIndex)

			// Finally, the partial derivative of error with respect to the output neuron weight is:
			// ∂TotalError/∂OutputNeuronWeight = ∂TotalError/∂TotalNetInputToOutputNeuron * ∂TotalNetInputToOutputNeuron/∂OutputNeuronWeight
			pdErrorWrtWeight := pdErrorWrtTotalNetInputOfOutputNeuron * pdTotalNetInputWrtWeight

			// Now that we know how much the output neuron's weight affects the error in the output, we get the new weight
			// by subtracting the affect from the current weight after multiplying it with the learning rate.
			// The learning rate is a constant value chosen for a network to control the correction in
			// a network's weight based on a sample.
			neuron.newWeights[weightIndex] = weight - (learningRate * pdErrorWrtWeight)
		}

		// By applying the chain rule, we can define the partial differential of total error with respect to the bias to the output neuron as:
		// ∂TotalError/∂OutputNeuronBias = ∂TotalError/∂TotalNetInputToOutputNeuron * ∂TotalNetInputToOutputNeuron/∂OutputNeuronBias
		//
		// Now, since the total net input of a neuron is a weighted summation of all the inputs and their respective weights to the neuron plus the bias of the neuron.
		// i.e. TotalNetInput = (n Σ ᵢ = 1) ((inputᵢ * weightᵢ) + biasᵢ)
		// The partial differential of total net input with respect to the bias is 1 since all other terms are treated as constants and bias doesn't has and multiplier.
		// Therefore,
		// ∂TotalError/∂OutputNeuronBias = ∂TotalError/∂TotalNetInputToOutputNeuron
		pdErrorWrtBias := pdErrorWrtTotalNetInputOfOutputNeuron

		// Now that we know how much the output neuron's bias affects the error in the output, we get the new bias weight
		// by subtracting the affect from the current bias after multiplying it with the learning rate.
		// The learning rate is a constant value chosen for a network to control the correction in
		// a network's bias based on a sample.
		neuron.newBias = neuron.bias - (learningRate * pdErrorWrtBias)
	}
}

// calculateNewHiddenLayerWeights function calculates new weights from the input
// layer to the hidden layer and bias for the hidden layer neurons, after
// calculating how much each weight and bias affects the error in the final
// output of the network. i.e. the partial differential of error with respect to
// the weight. ∂Error/∂Weight and the partial differential of error with respect
// to the bias. ∂Error/∂Bias.
//
// By applying the chain rule, https://en.wikipedia.org/wiki/Chain_rule
// ∂TotalError/∂HiddenNeuronWeight = ∂TotalError/∂HiddenNeuronOutput * ∂HiddenNeuronOutput/∂TotalNetInputToHiddenNeuron * ∂TotalNetInputToHiddenNeuron/∂HiddenNeuronWeight
func (network *NeuralNetwork) calculateNewHiddenLayerWeights() {
	// First we calculate the derivative of total error with respect to the output of each hidden neuron.
	// i.e. ∂TotalError/∂HiddenNeuronOutput.
	for neuronIndex, neuron := range network.hiddenLayer.neurons {
		// Since the total error is a summation of errors in each output neuron's output, we need to calculate the
		// derivative of error in each output neuron with respect to the output of each hidden neuron and add them.
		// i.e. ∂TotalError/∂HiddenNeuronOutput = ∂Error1/∂HiddenNeuronOutput + ∂Error2/∂HiddenNeuronOutput + ...
		dErrorWrtOutputOfHiddenNeuron := float64(0)
		for _, outputNeuron := range network.outputLayer.neurons {
			// The partial derivative of an output neuron's output's error with respect to the output of the hidden neuron can be expressed as:
			// ∂Error/∂HiddenNeuronOutput = ∂Error/∂TotalNetInputToOutputNeuron * ∂TotalNetInputToOutputNeuron/∂HiddenNeuronOutput
			//
			// We already have partial derivative of output neuron's error with respect to its total net input for each neuron from previous calculations.
			// Also, the partial derivative of total net input of output neuron with respect to the output of the current hidden neuron (∂TotalNetInputToOutputNeuron/∂HiddenNeuronOutput),
			// is the weight from the current hidden neuron to the current output neuron.
			dErrorWrtOutputOfHiddenNeuron += outputNeuron.pdErrorWrtTotalNetInputOfOutputNeuron * outputNeuron.weights[neuronIndex]
		}

		// We calculate the derivative of hidden neuron output with respect to total net input to hidden neuron,
		// ΔHiddenNeuronOutput/ΔTotalNetInputToHiddenNeuron
		dHiddenNeuronOutputWrtTotalNetInputToHiddenNeuron := neuron.calculateDerivativeOutputWrtTotalNetInput()

		// Next the partial derivative of error with respect to the total net input of the hidden neuron is:
		// ∂TotalError/∂TotalNetInputToHiddenNeuron = ∂TotalError/∂HiddenNeuronOutput * dHiddenNeuronOutput/dTotalNetInputToHiddenNeuron
		pdErrorWrtTotalNetInputOfHiddenNeuron := dErrorWrtOutputOfHiddenNeuron * dHiddenNeuronOutputWrtTotalNetInputToHiddenNeuron

		for weightIndex, weight := range neuron.weights {
			// For each weight of the neuron we calculate the partial derivative of
			// total net input with respect to the weight i.e. ∂TotalNetInputToHiddenNeuron/∂HiddenNeuronWeight
			pdTotalNetInputWrtWeight := neuron.calculatePdTotalNetInputWrtWeight(weightIndex)

			// Finally, the partial derivative of total error with respect to the hidden neuron weight is:
			// ∂TotalError/∂HiddenNeuronWeight = ∂TotalError/∂TotalNetInputToHiddenNeuron * ∂TotalNetInputToHiddenNeuron/∂HiddenNeuronWeight
			pdErrorWrtWeight := pdErrorWrtTotalNetInputOfHiddenNeuron * pdTotalNetInputWrtWeight

			// Now that we know how much the hidden neuron's weight affects the error in the output, we get the new weight
			// by subtracting the affect from the current weight after multiplying it with the learning rate.
			// The learning rate is a constant value chosen for a network to control the correction in
			// a network's weight based on a sample.
			neuron.newWeights[weightIndex] = weight - (learningRate * pdErrorWrtWeight)
		}

		// By applying the chain rule, we can define the partial differential of total error with respect to the bias to the hidden neuron as:
		// ∂TotalError/∂HiddenNeuronBias = ∂TotalError/∂TotalNetInputToHiddenNeuron * ∂TotalNetInputToHiddenNeuron/∂HiddenNeuronBias
		//
		// Now, since the total net input of a neuron is a weighted summation of all the inputs and their respective weights to the neuron plus the bias of the neuron.
		// i.e. TotalNetInput = (n Σ ᵢ = 1) ((inputᵢ * weightᵢ) + biasᵢ)
		// The partial differential of total net input with respect to the bias is 1 since all other terms are treated as constants and bias doesn't has and multiplier.
		// Therefore,
		// ∂TotalError/∂HiddenNeuronBias = ∂TotalError/∂TotalNetInputToHiddenNeuron
		pdErrorWrtBias := pdErrorWrtTotalNetInputOfHiddenNeuron

		// Now that we know how much the hidden neuron's bias affects the error in the output, we get the new bias weight
		// by subtracting the affect from the current bias after multiplying it with the learning rate.
		// The learning rate is a constant value chosen for a network to control the correction in
		// a network's bias based on a sample.
		neuron.newBias = neuron.bias - (learningRate * pdErrorWrtBias)
	}
}

// updateWeights updates the weights and biases for the hidden and output layer
// neurons with the new weights and biases.
func (network *NeuralNetwork) updateWeights() {
	for _, neuron := range network.outputLayer.neurons {
		neuron.weights = neuron.newWeights
		neuron.bias = neuron.newBias
	}

	for _, neuron := range network.hiddenLayer.neurons {
		neuron.weights = neuron.newWeights
		neuron.bias = neuron.newBias
	}
}

// CalculateTotalError computes and returns the total error for the given training set.
func (network *NeuralNetwork) CalculateTotalError(trainingSet [][][]float64) float64 {
	totalError := float64(0)
	for _, set := range trainingSet {
		output := network.CalculateOutput(set[0])
		_ = output // we don't need output here.
		totalError += network.CalculateError(set[1])
	}
	return totalError
}

// CalculateError function generates the error value for the given target output against the network's last output.
func (network *NeuralNetwork) CalculateError(targetOutput []float64) float64 {
	error := float64(0)
	for index, neuron := range network.outputLayer.neurons {
		error += neuron.calculateError(targetOutput[index])
	}
	return error
}

// Describe function prints the current state of the neural network and its components.
func (network *NeuralNetwork) Describe() {
	fmt.Println(fmt.Sprintf("Hidden Layer: (No of neurons: %v)", len(network.hiddenLayer.neurons)))
	network.hiddenLayer.describe()
	fmt.Println(fmt.Sprintf("\nOutput Layer: (No of neurons: %v)", len(network.outputLayer.neurons)))
	network.outputLayer.describe()
}
