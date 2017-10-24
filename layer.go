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
