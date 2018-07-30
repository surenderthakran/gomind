package activation

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

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
