package gomind

import (
	"fmt"
	"math"
)

type neuron struct {
	weights  []float64
	bias     float64
}

func (n *neuron) String() string {
	return fmt.Sprintf(`Neuron {
	weights: %v,
	bias: %v,
}`, n.weights, n.bias)
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

func (n *neuron) calculateOutput(inputs []float64) float64 {
	n.inputs = inputs
	n.netInput = n.calculateTotalNetInput(n.inputs)
	n.output = squash(n.netInput)
	return n.output
}

// calculateTotalNetInput function returns the final input to a neuron for an array of
// inputs based on its current set of weights.
//
// The total net input of a neuron is a weighted summation of all the inputs and their respective weights to the neuron plus the bias of the neuron.
// Total Net Input = (n Σ ᵢ = 1) ((inputᵢ * weightᵢ) + biasᵢ)
func (n *neuron) calculateTotalNetInput(input []float64) float64 {
	netInput := float64(0)
	for i := range input {
		netInput += input[i] * n.weights[i]
	}
	return netInput + n.bias
}

// squash function applies the non-linear sigmoid activation function on the total net input of a neuron to generate its output.
// f(x) = 1 * (1 + (e ^ -x))
func squash(input float64) float64 {
	// to avoid floating-point overflow in the exponential function, we use the
	// constant 45 as limiting value on the extremes.
	if input < -45 {
		return 0
	} else if input > 45 {
		return 1
	} else {
		return 1.0 / (1.0 + math.Exp(-input))
	}
}

func (n *neuron) calculateError(targetOutput float64) float64 {
	return 0.5 * math.Pow(targetOutput-n.output, 2)
}
