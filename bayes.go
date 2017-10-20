package bayes

type Label string

type FeatureSet map[string]string

type Probability float32

// Classifier Interface for a classifier that assignes a `class` or `label`
// to a an `entity` or `token` represented by one or more features.
type Classifier interface {
	Labels() []Label
	Classify(features []FeatureSet) Label
	ProbClassify(features []FeatureSet) ProbDist
}

type TrainFeatureSet struct {
	Label      Label
	FeatureSet FeatureSet
}

type ProbDist interface {
	Labels() []Label
	Prob(label Label) Probability
	Max() Label
}

type NaiveBayes struct {
}

func Train(fs []TrainFeatureSet) {
}
