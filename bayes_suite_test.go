package bayes_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestBayes(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Bayes Suite")
}
