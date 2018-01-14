// Package gomind for a simple Multi Layer Perceptron (MLP) Feed Forward Artificial Neural Network library.
package gomind

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/surenderthakran/gomind/activation"
	"github.com/surenderthakran/gomind/layer"
)

// NeuralNetwork describes a single hidden layer MLP feed forward neural network.
type NeuralNetwork struct {
	model        *ModelConfiguration
	learningRate float64
	hiddenLayer  *layer.Layer
	outputLayer  *layer.Layer
}

type ModelConfiguration struct {
	NumberOfInputs                    int
	NumberOfOutputs                   int
	ModelType                         string
	NumberOfHiddenLayerNeurons        int
	LearningRate                      float64
	HiddenLayerActivationFunctionName string
	OutputLayerActivationFunctionName string
}

func New(model *ModelConfiguration) (*NeuralNetwork, error) {
	fmt.Println("Initializing new Neural Network!")
	// setting timestamp as seed for random number generator.
	rand.Seed(time.Now().UnixNano())

	if model.NumberOfInputs == 0 {
		return nil, errors.New("NumberOfInputs field in ModelConfiguration is a mandatory field which cannot be zero.")
	}

	if model.NumberOfOutputs == 0 {
		return nil, errors.New("NumberOfOutputs field in ModelConfiguration is a mandatory field which cannot be zero.")
	}

	learningRate := 0.5
	if model.LearningRate != 0 {
		if model.LearningRate < 0 || model.LearningRate > 1 {
			return nil, errors.New("LearningRate cannot be less than 0 or greater than 1.")
		}
		learningRate = model.LearningRate
	}

	modelType := strings.Replace(strings.TrimSpace(strings.ToLower(model.ModelType)), " ", "", -1)
	if modelType == "regression" {
		fmt.Println("Configuring Neural network for Regression model")

		numberOfHiddenNeurons := model.NumberOfHiddenLayerNeurons
		if numberOfHiddenNeurons == 0 {
			numberOfHiddenNeurons = estimateIdealNumberOfHiddenLayerNeurons(model.NumberOfInputs, model.NumberOfOutputs)
			fmt.Println("Estimated Ideal Number Of Hidden Layer Neurons: ", numberOfHiddenNeurons)
		}

		hiddenLayerActivationFunctionName := model.HiddenLayerActivationFunctionName
		if hiddenLayerActivationFunctionName == "" {
			hiddenLayerActivationFunctionName = "LEAKY_RELU"
			fmt.Println("Estimated Ideal Activation Function for Hidden Layer Neurons: ", hiddenLayerActivationFunctionName)
		}
		hiddenLayerActivationService, err := activation.New(hiddenLayerActivationFunctionName)
		if err != nil {
			return nil, fmt.Errorf("invalid activation function: %v", err)
		}

		hiddenLayer, err := layer.New(numberOfHiddenNeurons, model.NumberOfInputs, hiddenLayerActivationService)
		if err != nil {
			return nil, fmt.Errorf("unable to create neural network: %v", err)
		}

		outputLayerActivationFunctionName := model.OutputLayerActivationFunctionName
		if outputLayerActivationFunctionName == "" {
			outputLayerActivationFunctionName = "SIGMOID"
			fmt.Println("Estimated Ideal Activation Function for Output Layer Neurons: ", outputLayerActivationFunctionName)
		}
		outputLayerActivationService, err := activation.New(outputLayerActivationFunctionName)
		if err != nil {
			return nil, fmt.Errorf("invalid activation function: %v", err)
		}

		outputLayer, err := layer.New(model.NumberOfOutputs, numberOfHiddenNeurons, outputLayerActivationService)
		if err != nil {
			return nil, fmt.Errorf("unable to create neural network: %v", err)
		}

		return &NeuralNetwork{
			model:        model,
			learningRate: learningRate,
			hiddenLayer:  hiddenLayer,
			outputLayer:  outputLayer,
		}, nil
	} else if modelType == "classification" {
		// TODO(surenderthakran): support auto-configuration for classification type neural network models.
		return nil, errors.New("We don't support classification type neural network models yet but hope to soon catch up on it.")
	}

	return nil, errors.New("invalid neural network model type. model should be either \"regresion\" or \"classification\".")
}

// estimateIdealNumberOfHiddenLayerNeurons function attempts to estimate the ideal number of neural networks in the hidden layer
// of the network for a given number of inputs and outputs.
func estimateIdealNumberOfHiddenLayerNeurons(numberOfInputs, numberOfOutputs int) int {
	var possibleResults []int
	twoThirdRule := ((numberOfInputs * 2) / 3) + numberOfOutputs
	possibleResults = append(possibleResults, twoThirdRule)
	if len(possibleResults) == 1 && possibleResults[0] < 2*numberOfInputs {
		if numberOfInputs < numberOfOutputs && numberOfInputs <= possibleResults[0] && possibleResults[0] <= numberOfOutputs {
			return possibleResults[0]
		} else if numberOfOutputs < numberOfInputs && numberOfOutputs <= possibleResults[0] && possibleResults[0] <= numberOfInputs {
			return possibleResults[0]
		} else if numberOfOutputs == numberOfInputs {
			return possibleResults[0]
		}
	}
	return numberOfInputs
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
			neuron.SetNewWeight(weight-(network.learningRate*pdErrorWrtWeight), weightIndex)
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
		neuron.SetNewBias(neuron.Bias() - (network.learningRate * pdErrorWrtBias))
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
			neuron.SetNewWeight(weight-(network.learningRate*pdErrorWrtWeight), weightIndex)
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
		neuron.SetNewBias(neuron.Bias() - (network.learningRate * pdErrorWrtBias))
	}
}

// updateWeights updates the weights and biases for the hidden and output layer
// neurons with the new weights and biases.
func (network *NeuralNetwork) updateWeights() {
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
	if math.IsInf(outputError, 1) || math.IsInf(outputError, -1) {
		return outputError, errors.New("error in the output is too high.")
	}
	if math.IsNaN(outputError) {
		return outputError, errors.New("error in the output is NaN.")
	}
	return outputError, nil
}

// Describe function prints the current state of the neural network and its components.
func (network *NeuralNetwork) Describe(showNeurons bool) {
	fmt.Println(fmt.Sprintf("Input Layer: (No of nodes: %v)", network.model.NumberOfInputs))
	fmt.Println(fmt.Sprintf("Hidden Layer: (No of neurons: %v, Activation Function: %v)", len(network.hiddenLayer.Neurons()), network.hiddenLayer.Activation().Name()))
	if showNeurons == true {
		network.hiddenLayer.Describe()
	}
	fmt.Println(fmt.Sprintf("Output Layer: (No of neurons: %v, Activation Function: %v))", len(network.outputLayer.Neurons()), network.outputLayer.Activation().Name()))
	if showNeurons == true {
		network.outputLayer.Describe()
	}
}
