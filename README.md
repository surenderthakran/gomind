# GoMind

## Installation
```
go get github.com/surenderthakran/gomind
```

## Usage
```
package main

import (
	"github.com/singhsurender/gomind"
)

func main() {
	trainingSet := [][][]float64{
		[][]float64{[]float64{0, 0}, []float64{0}},
		[][]float64{[]float64{0, 1}, []float64{1}},
		[][]float64{[]float64{1, 0}, []float64{1}},
		[][]float64{[]float64{1, 1}, []float64{0}},
	}

	mind, err := gomind.New(&gomind.ModelConfiguration{
		NumberOfInputs:                    2,
		NumberOfOutputs:                   1,
		NumberOfHiddenLayerNeurons:        16,
		HiddenLayerActivationFunctionName: "relu",
		OutputLayerActivationFunctionName: "sigmoid",
	})
	if err != nil {
		return nil, fmt.Errorf("unable to create neural network. %v", err)
	}

	for i := 0; i < 500; i++ {
		rand := rand.Intn(4)
		input := trainingSet[rand][0]
		output := trainingSet[rand][1]

		if err := mind.LearnSample(input, output); err != nil {
			mind.Describe(true)
			return nil, fmt.Errorf("error while learning from sample input: %v, target: %v. %v", input, output, err)
		}

		actual := mind.LastOutput()
		outputError, err := mind.CalculateError(output)
		if err != nil {
			mind.Describe(true)
			return nil, fmt.Errorf("error while calculating error for input: %v, target: %v and actual: %v. %v", input, output, actual, err)
		}

		outputAccuracy, err := mind.CalculateAccuracy(output)
		if err != nil {
			mind.Describe(true)
			return nil, fmt.Errorf("error while calculating error for input: %v, target: %v and actual: %v. %v", input, output, actual, err)
		}

		fmt.Println("Index: %v, Input: %v, Target: %v, Actual: %v, Error: %v, Accuracy: %v\n", i, input, output, actual, outputError, outputAccuracy)
	}
}
```

## API Documentation
The documentation for various methods exposed by the library can be found at: [https://godoc.org/github.com/surenderthakran/gomind](https://godoc.org/github.com/surenderthakran/gomind)
