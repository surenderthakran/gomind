package gomind

import (
	"fmt"
	"math/rand"

	"github.com/surenderthakran/gomind/neuron"
)

type layer struct {
	neurons []*neuron.Neuron
}

func newLayer(numberOfNeurons, numberOfNeuronsInPreviousLayer int) (*layer, error) {
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
	return &layer{
		neurons: neurons,
	}, nil
}

// calculateOutput function returns the output array from a layer of neurons for an
// array of input for the current set of weights of its neurons.
func (l *layer) calculateOutput(input []float64) []float64 {
	var output []float64
	for _, neuron := range l.neurons {
		output = append(output, neuron.CalculateOutput(input))
	}
	return output
}

func (l *layer) describe() {
	for index, neuron := range l.neurons {
		fmt.Println(fmt.Sprintf("Neuron: %v", index+1))
		fmt.Println(neuron)
	}
}
