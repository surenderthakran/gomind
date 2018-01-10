package layer

import (
	"fmt"
	"math/rand"

	"github.com/surenderthakran/gomind/neuron"
)

type Layer struct {
	neurons []*neuron.Neuron
}

func New(numberOfNeurons, numberOfNeuronsInPreviousLayer int) (*Layer, error) {
	if numberOfNeurons <= 0 {
		return nil, fmt.Errorf("%d is not a valid number of neurons", numberOfNeurons)
	}

	var neurons []*neuron.Neuron
	for i := 0; i < numberOfNeurons; i++ {
		var weights []float64
		for i := 0; i < numberOfNeuronsInPreviousLayer; i++ {
			weights = append(weights, rand.Float64())
		}

		bias := rand.Float64()

		neuron, err := neuron.New(weights, bias)
		if err != nil {
			return nil, fmt.Errorf("error creating a neuron: %v", err)
		}
		neurons = append(neurons, neuron)
	}
	return &Layer{
		neurons: neurons,
	}, nil
}

// Neurons function returns an array of pointers to neurons of the layer.
func (l *Layer) Neurons() []*neuron.Neuron {
	return l.neurons
}

// CalculateOutput function returns the output array from a layer of neurons for an
// array of input for the current set of weights of its neurons.
func (l *Layer) CalculateOutput(input []float64) []float64 {
	var output []float64
	for _, neuron := range l.neurons {
		output = append(output, neuron.CalculateOutput(input))
	}
	return output
}

// Describe function prints the description of the neurons in the layer.
func (l *Layer) Describe() {
	for index, neuron := range l.neurons {
		fmt.Println(fmt.Sprintf("Neuron: %v", index+1))
		fmt.Println(neuron)
	}
}
