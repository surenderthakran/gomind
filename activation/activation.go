package activation

import (
	"fmt"
	"strings"
)

var (
	activationFunctions = []string{"SIGMOID", "RELU", "LEAKY_RELU", "IDENTITY"}
)

type Service struct {
	name string
}

func New(name string) (*Service, error) {
	name = ValidFunction(name)
	if name != "" {
		return &Service{
			name: name,
		}, nil
	}
	return nil, fmt.Errorf("invalid activation function: %v. Activation Functions should be amongst: %v", name, activationFunctions)
}

func (s *Service) Name() string {
	return s.name
}

func SupportedActivationFunctions() []string {
	return activationFunctions
}

func ValidFunction(name string) string {
	name = strings.Replace(strings.TrimSpace(strings.ToUpper(name)), " ", "", -1)
	for _, functionName := range activationFunctions {
		if functionName == name {
			return functionName
		}
	}
	return ""
}
