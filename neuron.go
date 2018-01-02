package gomind

import (
	"fmt"
	"math"
)

type neuron struct {
	inputs     []float64
	weights    []float64
	newWeights []float64
	bias       float64
	newBias    float64
	netInput   float64
	output     float64
	// Holds the partial derivative of error with respect to the total net input.
	// This value is only relevant for the output layer neurons.
	pdErrorWrtTotalNetInputOfOutputNeuron float64
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
		weights:    weights,
		newWeights: make([]float64, len(weights)),
		bias:       bias,
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

// calculatePdErrorWrtTotalNetInputOfOutputNeuron function is only for output layer neurons.
// It returns the partial differential of output's error with respect to
// the total net input to the neuron. i.e. ∂Error/∂Input
//
// By applying the chain rule, https://en.wikipedia.org/wiki/Chain_rule
// ∂Error/∂Input = ∂Error/∂Output * ∂Output/∂Input
func (n *neuron) calculatePdErrorWrtTotalNetInputOfOutputNeuron(targetOutput float64) float64 {
	pdErrorWrtOutput := n.calculatePdErrorWrtOutput(targetOutput)
	dOutputWrtTotalNetInput := n.calculateDerivativeOutputWrtTotalNetInput()
	n.pdErrorWrtTotalNetInputOfOutputNeuron = pdErrorWrtOutput * dOutputWrtTotalNetInput
	return n.pdErrorWrtTotalNetInputOfOutputNeuron
}

// calculatePdErrorWrtOutput function is only for output layer neurons.
// It returns the partial derivative of a neuron's output's error with respect to its output.
//
// Error of a neuron's output is calculated from the Squared Error function.
// https://en.wikipedia.org/wiki/Backpropagation#Derivation
// Error = 1/2 * (target output - actual output) ^ 2
// The factor of 1/2 is included to cancel the exponent when differentiating.
//
// A partial differential of the error with respect to the actual output gives us:
// ∂Error/∂Actual = ∂(1/2 * (Target - Actual) ^ 2)/∂Actual
// 								= 1/2 * ∂((Target - Actual) ^ 2)/∂Actual
// 								= 1/2 * 2 * ((Target - Actual) ^ (2 - 1)) * ∂(Target - Actual)/∂Actual
// 								= 1/2 * 2 * ((Target - Actual) ^ (2 - 1)) * -1
// 								= - (Target - Actual)
// 								= Actual - Target
func (n *neuron) calculatePdErrorWrtOutput(targetOutput float64) float64 {
	return n.output - targetOutput
}

// calculateDerivativeOutputWrtTotalNetInput function is used by both hidden and output layer neurons.
// It returns the derivative (not partial derivative) of a neuron's output with respect to  the total net input.
// Since a neuron's total net input is squashed using the sigmoid function to get its output,
// we need to calculate the derivative of the sigmoid function.
// Output = 1.0 / (1.0 + (e ^ -Input))
//
// dOutput/dInput = d(1.0 / (1.0 + (e ^ -Input)))/dInput
// According to, https://en.wikipedia.org/wiki/Logistic_function#Derivative
// dOutput/dInput = Output * (1 - Output)
func (n *neuron) calculateDerivativeOutputWrtTotalNetInput() float64 {
	return n.output * (1 - n.output)
}

// calculatePdTotalNetInputWrtWeight function is used by both hidden and output layer neurons.
// It returns the partial derivative of total net input to a neuron with respect to one of its weight
// i.e. ∂TotalNetInput/∂Weight.
//
// The total net input of a neuron is a weighted summation of all the inputs and their respective weights to the neuron plus the bias of the neuron.
// Total Net Input = (n Σ ᵢ = 1) ((inputᵢ * weightᵢ) + biasᵢ)
//
// The partial derivative of the total net input with respect to the weight is the input for that particular weight
// since all the weighted sums and the bias are treated as constants.
func (n *neuron) calculatePdTotalNetInputWrtWeight(index int) float64 {
	return n.inputs[index]
}

func (n *neuron) calculateError(targetOutput float64) float64 {
	return 0.5 * math.Pow(targetOutput-n.output, 2)
}
