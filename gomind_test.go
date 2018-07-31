package gomind

import (
	"errors"
	"testing"

	"github.com/surenderthakran/gomind/layer"

	"github.com/google/go-cmp/cmp"
)

type fakeNeuralNetwork struct{}

func (fakeNeuralNetwork) CalculateOutput(input []float64) []float64 {
	return []float64{0}
}

func (fakeNeuralNetwork) LastOutput() []float64 {
	return []float64{0}
}

func (fakeNeuralNetwork) HiddenLayer() *layer.Layer {
	return nil
}

func (fakeNeuralNetwork) OutputLayer() *layer.Layer {
	return nil
}

func (fakeNeuralNetwork) CalculateNewOutputLayerWeights(outputs, targetOutputs []float64) error {
	return nil
}

func (fakeNeuralNetwork) CalculateNewHiddenLayerWeights() error {
	return nil
}

func (fakeNeuralNetwork) CalculateError(targetOutput []float64) (float64, error) {
	return 0, nil
}

func (fakeNeuralNetwork) UpdateWeights() {}

func TestModel(t *testing.T) {
	model := &Model{
		numberOfInputs:                    2,
		numberOfHiddenNeurons:             16,
		hiddenLayerActivationFunctionName: "RELU",
		numberOfOutputs:                   1,
		outputLayerActivationFunctionName: "SIGMOID",
		learningRate:                      0.5,
		network:                           &fakeNeuralNetwork{},
	}

	t.Run("LastOutput", func(t *testing.T) {
		lastOutput := model.LastOutput()

		if !cmp.Equal([]float64{0}, lastOutput) {
			t.Errorf("Model.LastOutput() = %d, want %d", lastOutput, []float64{0})
		}
	})
}

func TestEstimateIdealNumberOfHiddenLayerNeurons(t *testing.T) {
	testCases := []struct {
		inputs  int
		outputs int
		want    int
	}{
		{
			inputs:  2,
			outputs: 1,
			want:    2,
		},
		{
			inputs:  20,
			outputs: 1,
			want:    14,
		},
	}

	for _, test := range testCases {
		hidden := estimateIdealNumberOfHiddenLayerNeurons(test.inputs, test.outputs)

		if hidden != test.want {
			t.Errorf("estimateIdealNumberOfHiddenLayerNeurons(%d, %d) = %d, want %d", test.inputs, test.outputs, hidden, test.want)
		}
	}
}

func TestNew(t *testing.T) {
	testCases := []struct {
		modelConfig *ModelConfiguration
		model       *Model
		err         error
	}{
		{
			modelConfig: &ModelConfiguration{
				NumberOfInputs: 0,
			},
			err: errors.New("NumberOfInputs field in ModelConfiguration is a mandatory field which cannot be zero."),
		},
		{
			modelConfig: &ModelConfiguration{
				NumberOfInputs:  2,
				NumberOfOutputs: 0,
			},
			err: errors.New("NumberOfOutputs field in ModelConfiguration is a mandatory field which cannot be zero."),
		},
		{
			modelConfig: &ModelConfiguration{
				NumberOfInputs:  2,
				NumberOfOutputs: 1,
				LearningRate:    1.5,
			},
			err: errors.New("LearningRate cannot be less than 0 or greater than 1."),
		},
		{
			modelConfig: &ModelConfiguration{
				NumberOfInputs:  2,
				NumberOfOutputs: 1,
				LearningRate:    0.5,
			},
			err: nil,
		},
		{
			modelConfig: &ModelConfiguration{
				NumberOfInputs:                    2,
				NumberOfOutputs:                   1,
				NumberOfHiddenLayerNeurons:        5,
				LearningRate:                      0.5,
				HiddenLayerActivationFunctionName: "sigmoid",
				OutputLayerActivationFunctionName: "linear",
			},
			err: nil,
		},
	}

	for _, test := range testCases {
		_, err := New(test.modelConfig)

		if (err == nil && test.err != nil) || (err != nil && test.err == nil) || (err != nil && test.err != nil && err.Error() != test.err.Error()) {
			t.Errorf("New(%v) = _, %v, want _, %v", test.modelConfig, err, test.err)
		}
	}
}
