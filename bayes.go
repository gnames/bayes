package bayes

import (
	"errors"
	"fmt"
	"io"
)

// FeatureName is a name of a Feature
type FeatureName string

// FeatureName is a value of a Feature
type FeatureValue string

// Feature is an interface of an "evidence" we use for training a NaiveBayes
// classifier, or for the classification of an unknown entity.
type Featurer interface {
	// Name defines an id of a feature
	Name() FeatureName
	// Value defines the value of a feature. The value set can be simple
	// 'true|false' or more complex 'red|blue|grey|yellow'
	Value() FeatureValue
}

// FeatureFreq is a map for collecting frequencies of a training feature set.
// FeatureFreq is used for calculating Likelihoods of a NaiveBayes classifier.
type FeatureFreq map[FeatureName]map[FeatureValue]map[Labeler]float64

// FeatureTotal is used for calculating multinomial likelihoods. For example
// if we are interested in calculating likelihood of feature `f` its
// likelihood would be
//   L = P(f|H)/P(f/H')
// where `H` is main "hypothesis" or "label" and `H'` is a combination of all
// other hypotheses.
type FeatureTotal map[FeatureName]map[FeatureValue]float64

// LabelFreq is a collection of counts for every Label in the training dataset.
// this information allows to calculate prior odds for a Label.
type LabelFreq map[Labeler]float64

// LabeledFeatures are data used for supervised training of NaiveBayes
// algorithm.
type LabeledFeatures struct {
	Features []Featurer
	Label    Labeler
}

// Likelihoods provide a likelihood of a feature to appear for a
// particular label.
type Likelihoods map[Labeler]map[FeatureName]map[FeatureValue]float64

// OptionNB is a type for options supplied to NaiveBayes classifier. It can
// support either flags or parameterized options.
type OptionNB func(*NaiveBayes) error

// NewNaiveBayes is a constructor for NaiveBayes object. It initializes
// several important defaults, and sets options that modify behavior of
// the NaiveBayes object.
// Currently constructor supports the following options:
func NewNaiveBayes() *NaiveBayes {
	nb := &NaiveBayes{
		LabelFreq:       make(map[Labeler]float64),
		FeatureFreq:     make(map[FeatureName]map[FeatureValue]map[Labeler]float64),
		FeatureTotal:    make(map[FeatureName]map[FeatureValue]float64),
		IgnorePriorOdds: false,
	}
	return nb
}

// NaiveBayes is a classifier for assigning an entity represented by its
// features to a label.
type NaiveBayes struct {
	// Labels is a list of "hypotheses", "classes", "categories", "labels".
	// It contains all labels created by training.
	Labels []Labeler `json:"-"`
	// FeatureFreq keeps count of all the features for the labels.
	FeatureFreq `json:"-"`
	// LabelFreq keeps counts of the tokens belonging to each label
	LabelFreq `json:"-"`
	// FeatureTotal keeps total count of tokens for each feature.
	FeatureTotal `json:"feature_total"`
	// Total is a total number of tokens used for training.
	Total float64 `json:"total"`
	// IgnorePriorOdds is set true if prior odds do not need to be used in the
	// returning result.
	IgnorePriorOdds bool
	// currentLabelFreq keeps count for dynamically supplied labels frequencies
	currentLabelFreq LabelFreq
	// currentLabelTotal keeps total count of all supplied labels
	currentLabelTotal float64
	Output            io.Writer `json:"-"`
}

// TrainingPrior returns prior odds calculated from the training set
func (nb *NaiveBayes) TrainingPrior(l Labeler) (float64, error) {
	return Odds(l, nb.LabelFreq)
}

// Odds returns odds for a label in a given label frequency distribution
func Odds(l Labeler, lf LabelFreq) (float64, error) {
	var total, freq float64
	var ok bool
	if freq, ok = lf[l]; !ok {
		return 0.0, fmt.Errorf("Unkown label \"%s\"", l)
	}
	for _, v := range lf {
		total += v
	}
	pL := freq / total
	if pL == 1.0 || total == 0.0 {
		return 0.0, errors.New("Infinite prior odds")
	}
	return pL / (1 - pL), nil
}
