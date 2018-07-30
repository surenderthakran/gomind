package activation

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSupportedActivationFunctions(t *testing.T) {
	result := SupportedActivationFunctions()

	if !cmp.Equal(activationFunctions, result) {
		t.Errorf("SupportedActivationFunctions() = %v, want %v", activationFunctions, result)
	}
}
