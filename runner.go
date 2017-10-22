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

// WithPriorOdds allows to dynamically change prior odds used in calculations.
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
// trained set. It can take options that
func (nb *NaiveBayes) Predict(fs []Feature,
	opts ...OptionNB) (Posterior, error) {
	nb.currentLabelFreq = LabelFreq(nil)
	lf := nb.LabelFreq
	total := nb.total

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
	case l == 2:
		return pairPosterior(nb, fs, lf, total)
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

	for k1, _ := range nb.currentLabelFreq {
		if _, ok := nb.LabelFreq[k1]; !ok {
			return errors.New(
				fmt.Sprintf("Label %s does not exist in training set", k1))
		}
	}
	return nil
}

// special case when there are exactly 2 labels
func pairPosterior(nb *NaiveBayes, fs []Feature,
	lf LabelFreq, total float64) (Posterior, error) {
	label1 := nb.Labels[0]
	label2 := nb.Labels[1]

	lo := make(map[Label]float64)
	lo[label1] = oddsPrior(lf, label1, total)

	for _, f := range fs {
		p1 := nb.FeatureFreq[f.Name()][label1] /
			nb.LabelFreq[label1]
		p2 := nb.FeatureFreq[f.Name()][label2] /
			nb.LabelFreq[label2]
		lo[label1] *= p1 / p2
	}
	lo[label2] = 1 / lo[label1]

	maxL := label1
	maxO := lo[label1]
	if lo[label2] > maxO {
		maxL = label2
		maxO = lo[label2]
	}

	return Posterior{lo, maxL, maxO}, nil
}

func multiPosterior(nb *NaiveBayes, fs []Feature,
	lf LabelFreq, total float64) (Posterior, error) {
	var MaxLabel Label
	var MaxOdds float64
	lo := make(map[Label]float64)
	for _, label := range nb.Labels {
		lo[label] = oddsPrior(lf, label, total)
		for _, f := range fs {
			lo[label] *= likelihood(nb, f, label)
			if lo[label] > MaxOdds {
				MaxOdds = lo[label]
				MaxLabel = label
			}
		}
	}
	p := Posterior{lo, MaxLabel, MaxOdds}
	return p, nil
}

func likelihood(nb *NaiveBayes, feature Feature, label Label) float64 {
	name := feature.Name()
	featureFreq := nb.FeatureFreq[name][label]

	pFeature := featureFreq / nb.LabelFreq[label]

	pRest := (nb.featureTotal[name] - featureFreq) /
		(nb.total - nb.LabelFreq[label])
	return pFeature / pRest
}

func oddsPrior(lf LabelFreq, label Label, total float64) float64 {
	prob := lf[label] / total
	return prob / (1 - prob)
}
