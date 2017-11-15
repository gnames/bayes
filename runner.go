package bayes

import (
	"errors"
	"fmt"
)

// Posterior contains outcomes from NativeBayes classifier
type Posterior struct {
	LabelOdds map[Label]float64
	MaxLabel  Label
	MaxOdds   float64
}

// WithPriorOdds allows dynamical change of prior odds used in calculations.
// Sometimes prior odds during classification event are very different from
// ones aquired during training. If for example 'real' prior odds are 100 times
// larger it means the calculated posterior odds will be 100 times smaller than
// what they would suppose to be.
func WithPriorOdds(lf LabelFreq) func(*NaiveBayes) error {
	return func(nb *NaiveBayes) error {
		nb.currentLabelFreq = lf
		for _, v := range lf {
			nb.currentLabelTotal += v
		}
		return nil
	}
}

// Predict is a general function that runs NaiveBayes classifier against
// trained set. It can take a different PriorOdds value to influence
// calculation of the Posterior Odds.
func (nb *NaiveBayes) Predict(fs []Featurer,
	opts ...OptionNB) (Posterior, error) {
	nb.currentLabelFreq = LabelFreq(nil)
	lf := nb.LabelFreq
	total := nb.Total

	for _, o := range opts {
		err := o(nb)
		if err != nil {
			panic(err)
		}
	}

	if nb.currentLabelFreq != nil {
		lf = nb.currentLabelFreq
		total = nb.currentLabelTotal
		err := checkLabels(nb)
		if err != nil {
			return Posterior{}, err
		}
	}

	l := len(lf)
	switch {
	case l < 2:
		return Posterior{}, errors.New("Labels are empty")
	default:
		return multiPosterior(nb, fs, lf, total)
	}
}

func checkLabels(nb *NaiveBayes) error {
	err := errors.New("Empty or broken supplied prior odds")
	err2 := errors.New("Training odds differ in quantity from supplied odds")
	if nb.currentLabelFreq == nil || len(nb.currentLabelFreq) < 2 {
		return err
	} else if len(nb.currentLabelFreq) != len(nb.currentLabelFreq) {
		return err2
	}

	for k1 := range nb.currentLabelFreq {
		if _, ok := nb.LabelFreq[k1]; !ok {
			return fmt.Errorf("Label %s does not exist in training set", k1)
		}
	}
	return nil
}

func noSuchFeature(f Featurer, nb *NaiveBayes) bool {
	name := f.Name()
	if _, ok := nb.FeatureFreq[name]; ok {
		value := f.Value()
		if _, ok = nb.FeatureFreq[name][value]; ok {
			return false
		}
	}
	return true
}

func multiPosterior(nb *NaiveBayes, fs []Featurer,
	lf LabelFreq, total float64) (Posterior, error) {
	var MaxLabel Label
	var MaxOdds float64
	oddsPost := make(map[Label]float64)
	for _, label := range nb.Labels {
		oddsPost[label] = oddsPrior(lf, label, total)
		i := 0
		for _, f := range fs {
			if noSuchFeature(f, nb) {
				continue
			}
			i++
			oddsPost[label] *= likelihood(nb, f, label)
			if oddsPost[label] > MaxOdds {
				MaxOdds = oddsPost[label]
				MaxLabel = label
			}
		}
		if i == 0 {
			return Posterior{}, errors.New("All features are unknown")
		}
	}
	p := Posterior{oddsPost, MaxLabel, MaxOdds}
	return p, nil
}

func likelihood(nb *NaiveBayes, feature Featurer, label Label) float64 {
	smooth := 1.0
	name := feature.Name()
	value := feature.Value()
	countFeature := nb.FeatureFreq[name][value][label]
	countRest := (nb.FeatureTotal[name][value] - countFeature)
	pFeature := countFeature / nb.LabelFreq[label]

	// crude smoothing
	if countFeature == 0 {
		countFeature = smooth
	}
	if countRest == 0 {
		countRest = smooth
	}
	// end crude smoothing

	pRest := countRest / (nb.Total - nb.LabelFreq[label])
	return pFeature / pRest
}

func oddsPrior(lf LabelFreq, label Label, total float64) float64 {
	prob := lf[label] / total
	return prob / (1 - prob)
}
