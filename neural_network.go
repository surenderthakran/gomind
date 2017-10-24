// Package gomind for a simple Multi Layer Perceptron (MLP) Feed Forward Artificial Neural Network library.
package gomind

import (
	"fmt"
)

// NeuralNetwork describes a single hidden layer MLP feed forward neural network.
type NeuralNetwork struct {
	numberOfInputs int
	hiddenLayer    *layer
	outputLayer    *layer
}

// NewNeuralNetwork function returns a new NeuralNetwork object.
func NewNeuralNetwork(numberOfInputs, numberOfHiddenNeurons, numberOfOutputs int) (*NeuralNetwork, error) {
	fmt.Println(fmt.Sprintf("numberOfInputs: %d", numberOfInputs))

	fmt.Println(fmt.Sprintf("numberOfHiddenNeurons: %d", numberOfHiddenNeurons))
	hiddenLayer, err := newLayer(numberOfHiddenNeurons, numberOfInputs)
	if err != nil {
		return nil, fmt.Errorf("error creating a hidden layer: %v", err)
	}

	fmt.Println(fmt.Sprintf("numberOfOutputs: %d", numberOfOutputs))
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

// Describe function prints the current state of the neural network and its components.
func (network *NeuralNetwork) Describe() {
	fmt.Println("Hidden Layer:")
	network.hiddenLayer.describe()
	fmt.Println("\nOutput Layer:")
	network.outputLayer.describe()
}
