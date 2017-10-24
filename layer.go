package gomind

import (
	"fmt"
	"math/rand"
)

type layer struct {
	neurons []*neuron
}

func newLayer(numberOfNeurons, numberOfNeuronsInPreviousLayer int) (*layer, error) {
	if numberOfNeurons <= 0 {
		return nil, fmt.Errorf("%d is not a valid number of neurons", numberOfNeurons)
	}

	var neurons []*neuron
	for i := 0; i < numberOfNeurons; i++ {
		var weights []float64
		for i := 0; i < numberOfNeuronsInPreviousLayer; i++ {
			weights = append(weights, rand.Float64())
		}
		fmt.Println(fmt.Sprintf("weights: %v", weights))
		bias := rand.Float64()
		neuron, err := newNeuron(weights, bias)
		if err != nil {
			return nil, fmt.Errorf("error creating a neuron: %v", err)
		}
		neurons = append(neurons, neuron)
	}
	return &layer{
		neurons: neurons,
	}, nil
}

// calculateOutput function returns the output array from a layer of neurons for an
// array of input for the current set of weights of its neurons.
func (l *layer) calculateOutput(input []float64) []float64 {
	var output []float64
	for _, neuron := range l.neurons {
		output = append(output, neuron.calculateOutput(input))
	}
	return output
}

func (l *layer) describe() {
	fmt.Println("Neurons:")
	for _, neuron := range l.neurons {
		fmt.Println(neuron)
	}
}
