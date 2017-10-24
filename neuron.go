package gomind

import (
	"fmt"
)

type neuron struct {
	weights  []float64
	bias     float64
}

func newNeuron(weights []float64, bias float64) (*neuron, error) {
	if len(weights) == 0 {
		return nil, fmt.Errorf("unable to create neuron without any weights")
	}
	return &neuron{
		weights: weights,
		bias:    bias,
	}, nil
}
