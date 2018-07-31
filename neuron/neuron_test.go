package neuron

import (
	"fmt"
	"math"
	"testing"

	"github.com/surenderthakran/gomind/activation"

	"github.com/google/go-cmp/cmp"
)

func TestNeuron(t *testing.T) {
	activationFunction := "sigmoid"
	activationService, err := activation.New(activationFunction)
	if err != nil {
		t.Fatalf("activation.New(%s) -> %v", activationFunction, err)
	}

	neuron := &Neuron{
		weights:    []float64{0, 1},
		bias:       2,
		newWeights: []float64{0, 0},
		activation: activationService,
	}

	t.Run("String", func(t *testing.T) {
		want := `Neuron {
	weights: [0 1],
	bias: 2,
	activation: SIGMOID,
}
`

		result := neuron.String()

		if result != want {
			t.Errorf("Neuron.String() = %s, want: %s", result, want)
		}
	})

	t.Run("CalculateOutput", func(t *testing.T) {
		testCases := []struct {
			inputs []float64
			want   float64
		}{
			{
				inputs: []float64{0, 1},
				want:   0.95,
			},
		}

		for _, test := range testCases {
			output := neuron.CalculateOutput(test.inputs)
			output = roundTo(output, 2)

			if output != test.want {
				t.Errorf("Neuron.CalculateOutput(%v) = %f, want: %f", test.inputs, output, test.want)
			}
		}
	})
}

func TestNew(t *testing.T) {
	activationFunction := "sigmoid"
	activationService, err := activation.New(activationFunction)
	if err != nil {
		t.Fatalf("activation.New(%s) -> %v", activationFunction, err)
	}

	testCases := []struct {
		weights           []float64
		bias              float64
		activationService *activation.Service
		neuron            *Neuron
		err               error
	}{
		{
			weights:           []float64{},
			bias:              4,
			activationService: activationService,
			neuron:            nil,
			err:               fmt.Errorf("unable to create neuron without any weights"),
		},
		{
			weights:           []float64{0, 1},
			bias:              2,
			activationService: activationService,
			neuron: &Neuron{
				weights:    []float64{0, 1},
				bias:       2,
				newWeights: []float64{0, 0},
				activation: activationService,
			},
			err: nil,
		},
	}

	for _, test := range testCases {
		neuron, err := New(test.weights, test.bias, activationService)
		if (err == nil && test.err != nil) || (err != nil && test.err == nil) || (err != nil && test.err != nil && err.Error() != test.err.Error()) ||
			!cmp.Equal(test.neuron, neuron, cmp.AllowUnexported(Neuron{}, activation.Service{})) {
			t.Errorf("New(%v, %f, %v) = %v, %v, \nwant: %v, %v", test.weights, test.bias, activationService, neuron, err, test.neuron, test.err)
		}
	}
}

func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

func roundTo(input float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(input*output)) / output
}
