package bayes

import (
	"github.com/gnames/bayes/ent/bayesdump"
	ft "github.com/gnames/bayes/ent/feature"
	"github.com/gnames/bayes/ent/posterior"
)

// Trainer interface provides methods for training Bayes object to
// data from the training set.
type Trainer interface {
	Train([]ft.ClassFeatures)
}

// Serializer provides methods for dumping data from Bayes object to
// a slice of bytes, and rebuilding Bayes object from such data.
type Serializer interface {
	// Inspect returns back simplified and publicly accessed information that
	// is normally private for Bayes object.
	Inspect() bayesdump.BayesDump
	// Load takes a slice of bytes that corresponds to output.Output and
	// creates a Bayes instance from it.
	Load([]byte) error
	// Dump takes an internal data of a Bayes instance, converts it to
	// object.Object and serializes it to slice of bytes.
	Dump() ([]byte, error)
}

// Calc provides methods for calculating Prior and Posterior Odds from
// new data, allowing to classify the data according to its features.
type Calc interface {
	// PriorOdds method returns Odds from the training.
	PriorOdds(ft.Class) (float64, error)
	// PosteriorOdds uses set of features to determing which class they belong
	// to with the most probability.
	PosteriorOdds([]ft.Feature, ...Option) (posterior.Odds, error)
	// Likelihood gives an isolated likelihood of a feature.
	Likelihood(ft.Feature, ft.Class) (float64, error)
}

// Bayes interface uses Bayes algorithm for calculation of the posterior and
// prior odds. For training it takes manually curated data packed into
// features, and allows to serialize and deserialize the data.
type Bayes interface {
	Trainer
	Serializer
	Calc
}
