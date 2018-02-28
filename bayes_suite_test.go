package bayes_test

import (
	. "github.com/gnames/bayes"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestBayes(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Bayes Suite")
}

type Label int

const (
	Jar1 Label = iota
	Jar2
	Jar3
	Helicopter
	Surething
	Lots
	Little
)

var labels = []string{"Jar1", "Jar2", "Jar3", "Helicopter", "Surething", "Lots",
	"Little"}

var labelsDict = func() map[string]Labeler {
	res := make(map[string]Labeler)
	for i, v := range labels {
		res[v] = Label(i)
	}
	return res
}()

func (l Label) String() string {
	return labels[l]
}

var _ = BeforeSuite(func() {
	RegisterLabel(labelsDict)
})
