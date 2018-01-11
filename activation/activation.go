package activation

import "fmt"

type Name int

const (
	SIGMOID Name = iota
)

type Service struct {
	name Name
}

func New(name Name) (*Service, error) {
	switch name {
	case SIGMOID:
		return &Service{
			name: name,
		}, nil
	}
	return nil, fmt.Errorf("invalid activation function: %v", name)
}

func (s *Service) Name() Name {
	return s.name
}
