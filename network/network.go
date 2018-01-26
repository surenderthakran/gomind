package network

import (
	"fmt"
	"math"

	"github.com/surenderthakran/gomind/activation"
	"github.com/surenderthakran/gomind/layer"
)

// NeuralNetwork describes a single hidden layer MLP feed forward neural network.
type NeuralNetwork struct {
	learningRate float64
	hiddenLayer  *layer.Layer
	outputLayer  *layer.Layer
}

// CalculateOutput function returns the output array from the neural network for the given
// input array based on the current weights.
func (network *NeuralNetwork) CalculateOutput(input []float64) []float64 {
	hiddenOutput := network.hiddenLayer.CalculateOutput(input)
	return network.outputLayer.CalculateOutput(hiddenOutput)
}

// LastOutput function returns the array of last output computed by the network.
func (network *NeuralNetwork) LastOutput() []float64 {
	var output []float64
	for _, neuron := range network.outputLayer.Neurons() {
		output = append(output, neuron.Output())
	}
	return output
}

func (network *NeuralNetwork) HiddenLayer() *layer.Layer {
	return network.hiddenLayer
}

func (network *NeuralNetwork) OutputLayer() *layer.Layer {
	return network.outputLayer
}

// CalculateNewOutputLayerWeights function calculates new weights from the
// hidden layer to the output layer and bias for the output layer neurons, after
// calculating how much each weight and bias affects the total error in the
// final output of the network. i.e. the partial differential of error with
// respect to the weight. ∂Error/∂Weight and the partial differential of error
// with respect to the bias. ∂Error/∂Bias.
//
// By applying the chain rule, https://en.wikipedia.org/wiki/Chain_rule
// ∂TotalError/∂OutputNeuronWeight = ∂TotalError/∂TotalNetInputToOutputNeuron * ∂TotalNetInputToOutputNeuron/∂OutputNeuronWeight
func (network *NeuralNetwork) CalculateNewOutputLayerWeights(outputs, targetOutputs []float64) error {
	for neuronIndex, neuron := range network.outputLayer.Neurons() {
		// Since a neuron has only one total net input and one output, we need to calculate
		// the partial derivative of error with respect to the total net input (∂TotalError/∂TotalNetInputToOutputNeuron) only once.
		//
		// The total error of the network is a sum of errors in all the output neurons.
		// ex: Total Error = error1 + erro2 + error3 + ...
		// But when calculating the partial derivative of the total error with respect to the total net input
		// of only one output neuron, we need to find partial derivative of only the corresponding neuron's error because
		// the errors due to other neurons would be constant for it and their derivative wouldn't matter.
		pdErrorWrtTotalNetInputOfOutputNeuron := neuron.CalculatePdErrorWrtTotalNetInputOfOutputNeuron(targetOutputs[neuronIndex])

		for weightIndex, weight := range neuron.Weights() {
			// For each weight of the neuron we calculate the partial derivative of
			// total net input with respect to the weight i.e. ∂TotalNetInputToOutputNeuron/∂OutputNeuronWeight.
			pdTotalNetInputWrtWeight := neuron.CalculatePdTotalNetInputWrtWeight(weightIndex)

			// Finally, the partial derivative of error with respect to the output neuron weight is:
			// ∂TotalError/∂OutputNeuronWeight = ∂TotalError/∂TotalNetInputToOutputNeuron * ∂TotalNetInputToOutputNeuron/∂OutputNeuronWeight
			pdErrorWrtWeight := pdErrorWrtTotalNetInputOfOutputNeuron * pdTotalNetInputWrtWeight

			// Now that we know how much the output neuron's weight affects the error in the output, we get the new weight
			// by subtracting the affect from the current weight after multiplying it with the learning rate.
			// The learning rate is a constant value chosen for a network to control the correction in
			// a network's weight based on a sample.
			newWeight := weight - (network.learningRate * pdErrorWrtWeight)
			if math.IsInf(newWeight, 1) || math.IsInf(newWeight, -1) || math.IsNaN(newWeight) {
				return fmt.Errorf("invalid new weight: %v for output layer neuron.", newWeight)
			}
			neuron.SetNewWeight(newWeight, weightIndex)
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
		newBias := neuron.Bias() - (network.learningRate * pdErrorWrtBias)
		if math.IsInf(newBias, 1) || math.IsInf(newBias, -1) || math.IsNaN(newBias) {
			return fmt.Errorf("invalid new bias: %v for output layer neurons.", newBias)
		}
		neuron.SetNewBias(newBias)
	}
	return nil
}

// CalculateNewHiddenLayerWeights function calculates new weights from the input
// layer to the hidden layer and bias for the hidden layer neurons, after
// calculating how much each weight and bias affects the error in the final
// output of the network. i.e. the partial differential of error with respect to
// the weight. ∂Error/∂Weight and the partial differential of error with respect
// to the bias. ∂Error/∂Bias.
//
// By applying the chain rule, https://en.wikipedia.org/wiki/Chain_rule
// ∂TotalError/∂HiddenNeuronWeight = ∂TotalError/∂HiddenNeuronOutput * ∂HiddenNeuronOutput/∂TotalNetInputToHiddenNeuron * ∂TotalNetInputToHiddenNeuron/∂HiddenNeuronWeight
func (network *NeuralNetwork) CalculateNewHiddenLayerWeights() error {
	// First we calculate the derivative of total error with respect to the output of each hidden neuron.
	// i.e. ∂TotalError/∂HiddenNeuronOutput.
	for neuronIndex, neuron := range network.hiddenLayer.Neurons() {
		// Since the total error is a summation of errors in each output neuron's output, we need to calculate the
		// derivative of error in each output neuron with respect to the output of each hidden neuron and add them.
		// i.e. ∂TotalError/∂HiddenNeuronOutput = ∂Error1/∂HiddenNeuronOutput + ∂Error2/∂HiddenNeuronOutput + ...
		dErrorWrtOutputOfHiddenNeuron := float64(0)
		for _, outputNeuron := range network.outputLayer.Neurons() {
			// The partial derivative of an output neuron's output's error with respect to the output of the hidden neuron can be expressed as:
			// ∂Error/∂HiddenNeuronOutput = ∂Error/∂TotalNetInputToOutputNeuron * ∂TotalNetInputToOutputNeuron/∂HiddenNeuronOutput
			//
			// We already have partial derivative of output neuron's error with respect to its total net input for each neuron from previous calculations.
			// Also, the partial derivative of total net input of output neuron with respect to the output of the current hidden neuron (∂TotalNetInputToOutputNeuron/∂HiddenNeuronOutput),
			// is the weight from the current hidden neuron to the current output neuron.
			dErrorWrtOutputOfHiddenNeuron += outputNeuron.PdErrorWrtTotalNetInputOfOutputNeuron * outputNeuron.Weight(neuronIndex)
		}

		// We calculate the derivative of hidden neuron output with respect to total net input to hidden neuron,
		// ΔHiddenNeuronOutput/ΔTotalNetInputToHiddenNeuron
		dHiddenNeuronOutputWrtTotalNetInputToHiddenNeuron := neuron.CalculateDerivativeOutputWrtTotalNetInput()

		// Next the partial derivative of error with respect to the total net input of the hidden neuron is:
		// ∂TotalError/∂TotalNetInputToHiddenNeuron = ∂TotalError/∂HiddenNeuronOutput * dHiddenNeuronOutput/dTotalNetInputToHiddenNeuron
		pdErrorWrtTotalNetInputOfHiddenNeuron := dErrorWrtOutputOfHiddenNeuron * dHiddenNeuronOutputWrtTotalNetInputToHiddenNeuron

		for weightIndex, weight := range neuron.Weights() {
			// For each weight of the neuron we calculate the partial derivative of
			// total net input with respect to the weight i.e. ∂TotalNetInputToHiddenNeuron/∂HiddenNeuronWeight
			pdTotalNetInputWrtWeight := neuron.CalculatePdTotalNetInputWrtWeight(weightIndex)

			// Finally, the partial derivative of total error with respect to the hidden neuron weight is:
			// ∂TotalError/∂HiddenNeuronWeight = ∂TotalError/∂TotalNetInputToHiddenNeuron * ∂TotalNetInputToHiddenNeuron/∂HiddenNeuronWeight
			pdErrorWrtWeight := pdErrorWrtTotalNetInputOfHiddenNeuron * pdTotalNetInputWrtWeight

			// Now that we know how much the hidden neuron's weight affects the error in the output, we get the new weight
			// by subtracting the affect from the current weight after multiplying it with the learning rate.
			// The learning rate is a constant value chosen for a network to control the correction in
			// a network's weight based on a sample.
			newWeight := weight - (network.learningRate * pdErrorWrtWeight)
			if math.IsInf(newWeight, 1) || math.IsInf(newWeight, -1) || math.IsNaN(newWeight) {
				return fmt.Errorf("invalid new weight: %v for hidden layer neuron.", newWeight)
			}
			neuron.SetNewWeight(newWeight, weightIndex)
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
		newBias := neuron.Bias() - (network.learningRate * pdErrorWrtBias)
		if math.IsInf(newBias, 1) || math.IsInf(newBias, -1) || math.IsNaN(newBias) {
			return fmt.Errorf("invalid new bias: %v for hidden layer neurons.", newBias)
		}
		neuron.SetNewBias(newBias)
	}
	return nil
}

// UpdateWeights updates the weights and biases for the hidden and output layer
// neurons with the new weights and biases.
func (network *NeuralNetwork) UpdateWeights() {
	for _, neuron := range network.outputLayer.Neurons() {
		neuron.UpdateWeightsAndBias()
	}

	for _, neuron := range network.hiddenLayer.Neurons() {
		neuron.UpdateWeightsAndBias()
	}
}

// CalculateError function generates the error value for the given target output against the network's last output.
func (network *NeuralNetwork) CalculateError(targetOutput []float64) (float64, error) {
	outputError := float64(0)
	for index, neuron := range network.outputLayer.Neurons() {
		outputError += neuron.CalculateError(targetOutput[index])
	}
	outputError = outputError / float64(len(network.outputLayer.Neurons()))
	if math.IsInf(outputError, 1) || math.IsInf(outputError, -1) || math.IsNaN(outputError) {
		return outputError, fmt.Errorf("invalid error value: %v in output.", outputError)
	}
	return outputError, nil
}

func New(numberOfInputs, numberOfHiddenNeurons, numberOfOutputs int, learningRate float64, hiddenLayerActivationFunctionName, outputLayerActivationFunctionName string) (*NeuralNetwork, error) {
	hiddenLayerActivationService, err := activation.New(hiddenLayerActivationFunctionName)
	if err != nil {
		return nil, fmt.Errorf("invalid activation function: %v", err)
	}
	hiddenLayer, err := layer.New(numberOfHiddenNeurons, numberOfInputs, hiddenLayerActivationService)
	if err != nil {
		return nil, fmt.Errorf("unable to create hidden layer: %v", err)
	}

	outputLayerActivationService, err := activation.New(outputLayerActivationFunctionName)
	if err != nil {
		return nil, fmt.Errorf("invalid activation function: %v", err)
	}
	outputLayer, err := layer.New(numberOfOutputs, numberOfHiddenNeurons, outputLayerActivationService)
	if err != nil {
		return nil, fmt.Errorf("unable to create output layer: %v", err)
	}

	return &NeuralNetwork{
		learningRate: learningRate,
		hiddenLayer:  hiddenLayer,
		outputLayer:  outputLayer,
	}, nil
}
