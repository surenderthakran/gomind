<img src="https://golang.org/doc/gopher/fiveyears.jpg" width=800><br>

# GoMind

[![GoDoc](https://godoc.org/github.com/surenderthakran/gomind?status.png)](https://godoc.org/github.com/surenderthakran/gomind)
[![Go Report Card](https://goreportcard.com/badge/github.com/surenderthakran/gomind)](https://goreportcard.com/report/github.com/surenderthakran/gomind)
[![Release](https://img.shields.io/github/tag/surenderthakran/gomind.svg?label=latest)](https://github.com/surenderthakran/gomind/releases/tag/v1.0)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/singhsurender/gomind/blob/master/LICENSE)

## Installation
```
go get github.com/surenderthakran/gomind
```

## About
GoMind is a neural network library written entirely in Go.
It only supports a single hidden layer (for now).
The network learns from a training set using back-propagation algorithm.

Some of the salient features of GoMind are:
- Supports following activation functions:
  - Linear (Default)
  - [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
  - [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
  - [Leaky ReLU](https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29#Leaky_ReLUs)
- Smartly estimates ideal number of hidden layer neurons (if a count is not given during model configuration) for given input and output sizes.
- Uses [Mean Squared Error function](https://en.wikipedia.org/wiki/Mean_squared_error) to calculate error while back propagating.

Note: To understand the basic functioning of back-propagation in neural networks, one can refer to my blog [here](https://www.surenderthakran.com/articles/tech/implement-back-propagation-neural-network).

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
