package activation

import (
	"fmt"
	"strings"
)

var (
	activationFunctions = []string{"SIGMOID", "RELU", "LEAKY_RELU", "LINEAR"}
)

// Service type defines a new activation service for the GoMind neural network.
type Service struct {
	name string
}

// New creates a new activation service of the given activation function name.
func New(name string) (*Service, error) {
	name = ValidFunction(name)
	if name != "" {
		return &Service{
			name: name,
		}, nil
	}
	return nil, fmt.Errorf("invalid activation function: %v. Activation Functions should be amongst: %v", name, activationFunctions)
}

// Name returns the string name of the activation service.
func (s *Service) Name() string {
	return s.name
}

// SupportedActivationFunctions returns the list of activation functions supported by GoMind.
func SupportedActivationFunctions() []string {
	return activationFunctions
}

// ValidFunction returns the valid GoMind name of the given activation function's name.
func ValidFunction(name string) string {
	name = strings.Replace(strings.TrimSpace(strings.ToUpper(name)), " ", "", -1)
	for _, functionName := range activationFunctions {
		if functionName == name {
			return functionName
		}
	}
	return ""
}
