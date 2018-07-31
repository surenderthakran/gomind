package layer

import (
	"fmt"
	"math"
	"testing"

	"github.com/surenderthakran/gomind/activation"
	"github.com/surenderthakran/gomind/neuron"

	"github.com/google/go-cmp/cmp"
)

func TestLayer(t *testing.T) {
	activationFunction := "sigmoid"
	activationService, err := activation.New(activationFunction)
	if err != nil {
		t.Fatalf("activation.New(%s) -> %v", activationFunction, err)
	}

	weights1 := []float64{0.4, 0.6}
	bias1 := 0.5
	neuron1, err := neuron.New(weights1, bias1, activationService)
	if err != nil {
		t.Fatalf("neuron.New(%v, %f, %v) -> %v", weights1, bias1, activationFunction, err)
	}

	layer := &Layer{
		neurons: []*neuron.Neuron{
			neuron1,
		},
		activation: activationService,
	}

	t.Run("Neurons", func(t *testing.T) {
		neurons := layer.Neurons()

		if len(neurons) != 1 {
			t.Errorf("len(Layer.Neurons()) = %d, want %d", len(neurons), 1)
		}
	})

	t.Run("Activation", func(t *testing.T) {
		activationService := layer.Activation()

		if activationService.Name() != "SIGMOID" {
			t.Errorf("Layer.Activation().Name() = %s, want %s", activationService.Name(), "SIGMOID")
		}
	})

	t.Run("CalculateOutput", func(t *testing.T) {
		inputs := []float64{0, 1}
		outputs := layer.CalculateOutput(inputs)

		outputs[0] = roundTo(outputs[0], 2)

		if !cmp.Equal([]float64{0.75}, outputs) {
			t.Errorf("Layer.CalculateOutput(%v) = %v, want %v", inputs, outputs, []float64{0})
		}
	})
}

func TestNew(t *testing.T) {
	activationService := &activation.Service{}

	testCases := []struct {
		numberOfNeurons                int
		numberOfNeuronsInPreviousLayer int
		err                            error
	}{
		{
			numberOfNeurons:                0,
			numberOfNeuronsInPreviousLayer: 2,
			err: fmt.Errorf("0 is not a valid number of neurons"),
		},
		{
			numberOfNeurons:                2,
			numberOfNeuronsInPreviousLayer: 0,
			err: fmt.Errorf("error creating layer: \nunable to create neuron without any weights"),
		},
		{
			numberOfNeurons:                2,
			numberOfNeuronsInPreviousLayer: 1,
			err: nil,
		},
	}

	for _, test := range testCases {
		_, err := New(test.numberOfNeurons, test.numberOfNeuronsInPreviousLayer, activationService)
		if (err == nil && test.err != nil) || (err != nil && test.err == nil) || (err != nil && test.err != nil && err.Error() != test.err.Error()) {
			t.Errorf("New(%d, %d, %v) = _, %v, want _, %v", test.numberOfNeurons, test.numberOfNeuronsInPreviousLayer, activationService, err, test.err)
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
