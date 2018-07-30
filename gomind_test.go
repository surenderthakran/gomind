package gomind

import (
	"errors"
	"testing"
)

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
				LearningRate:    0,
			},
			err: errors.New("LearningRate cannot be less than or equals to 0 or greater than 1."),
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
