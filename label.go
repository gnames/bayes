package bayes

import "fmt"

// Labeler is an interface to represent a "hypothesis" or "class" the NaiveBayes
// is aware about.
type Labeler interface {
	fmt.Stringer
}

// labelMap is a private variable that keeps conversion from a string to
// a label. It can be set via RegisterLabel function.
var labelMap map[string]Labeler

// RegisterLabel takes a map from a string to Labeler interface. This map
// is required to unmarshal JSON data from a string to a label using
// LabelFactory.
func RegisterLabel(m map[string]Labeler) {
	labelMap = m
}

// LabelFactory takes a string and returns a Label. This function is mostly
// used for unmarshalling data from JSON to NaiveBayes object.
func LabelFactory(s string) (Labeler, error) {
	if l, ok := labelMap[s]; !ok {
		return l, fmt.Errorf("Cannot generate label from string '%s'.", s)
	} else {
		return l, nil
	}
}
