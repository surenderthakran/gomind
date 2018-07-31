package activation

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestNew(t *testing.T) {
	testCases := []struct {
		name    string
		service *Service
		err     error
	}{
		{
			name: "identity",
			err:  fmt.Errorf("invalid activation function: . Activation Functions should be amongst: %v", activationFunctions),
		},
		{
			name: "sigmoid",
			service: &Service{
				name: "SIGMOID",
			},
		},
	}

	for _, test := range testCases {
		service, err := New(test.name)
		if err != nil && err.Error() != test.err.Error() {
			t.Errorf("New(%s) = _, %v, want _, %v", test.name, err, test.err)
		} else if !cmp.Equal(test.service, service, cmp.AllowUnexported(Service{})) {
			t.Errorf("New(%s) = %v, nil, want %v, nil", test.name, service, test.service)
		}
	}
}

func TestName(t *testing.T) {
	service, _ := New("sigmoid")

	name := service.Name()

	if name != "SIGMOID" {
		t.Errorf("Service.Name() = %v, want %v", name, "SIGMOID")
	}
}

func TestSupportedActivationFunctions(t *testing.T) {
	result := SupportedActivationFunctions()

	if !cmp.Equal(activationFunctions, result) {
		t.Errorf("SupportedActivationFunctions() = %v, want %v", result, activationFunctions)
	}
}

func TestValidFunction(t *testing.T) {
	testCases := []struct {
		input string
		want  string
	}{
		{
			input: "sigmoid",
			want:  "SIGMOID",
		},
		{
			input: "identity",
			want:  "",
		},
		{
			input: "leaky relu",
			want:  "",
		},
	}

	for _, test := range testCases {
		result := ValidFunction(test.input)

		if result != test.want {
			t.Errorf("ValidFunction(%s) = %s, want %s", test.input, result, test.want)
		}
	}
}
