package bayes

import (
	"errors"
	"fmt"
)

// Posterior contains outcomes from NativeBayes classifier
type Posterior struct {
	LabelOdds map[Labeler]float64
	MaxLabel  Labeler
	MaxOdds   float64
	LabelFreq
	Likelihoods
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

// IgnorePriorOdds might be needed if it is a multistep Bayes calculation and
// PriorOdds already are accounted for.
func IgnorePriorOdds(nb *NaiveBayes) error {
	nb.IgnorePriorOdds = true
	return nil
}

// Predict is a general function that runs NaiveBayes classifier against
// trained set. It can take a different PriorOdds value to influence
// calculation of the Posterior Odds.
func (nb *NaiveBayes) Predict(fs []Featurer,
	opts ...OptionNB) (Posterior, error) {
	nb.currentLabelFreq = LabelFreq(nil)
	nb.IgnorePriorOdds = false
	lf := nb.LabelFreq

	for _, o := range opts {
		err := o(nb)
		if err != nil {
			panic(err)
		}
	}

	cf := nb.currentLabelFreq

	if cf != nil && validLabelFreq(cf) {
		lf = nb.currentLabelFreq
	}

	l := len(lf)
	if l < 2 {
		return Posterior{}, errors.New("Labels are empty")
	}
	pr, err := multiPosterior(nb, fs, lf)
	return pr, err
}

func validLabelFreq(lf LabelFreq) bool {
	var count int
	for _, v := range lf {
		if v > 0 {
			count++
			if count > 1 {
				return true
			}
		}
	}
	return false
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
	lf LabelFreq) (Posterior, error) {
	var MaxLabel Labeler
	var MaxOdds float64
	oddsPost := make(map[Labeler]float64)
	likelihoods := make(Likelihoods)
	for _, label := range nb.Labels {
		odds, err := Odds(label, lf)
		if err != nil {
			return Posterior{}, fmt.Errorf("Cannot calculate odds: %s", err.Error())
		}
		oddsPost[label] = 1
		if !nb.IgnorePriorOdds {
			oddsPost[label] = odds
		}
		likelihoods[label] = make(map[FeatureName]map[FeatureValue]float64)
		if !nb.IgnorePriorOdds {
			likelihoods[label][FeatureName("PriorOdds")] =
				map[FeatureValue]float64{FeatureValue("true"): odds}
		}

		i := 0
		for _, f := range fs {
			if noSuchFeature(f, nb) {
				continue
			}
			lh := likelihood(nb, f, label)
			likelihoods[label][f.Name()] = map[FeatureValue]float64{f.Value(): lh}

			i++
			oddsPost[label] *= lh
			if oddsPost[label] > MaxOdds {
				MaxOdds = oddsPost[label]
				MaxLabel = label
			}
		}
		if i == 0 {
			return Posterior{}, errors.New("All features are unknown")
		}
	}
	p := Posterior{oddsPost, MaxLabel, MaxOdds, lf, likelihoods}
	return p, nil
}

func likelihood(nb *NaiveBayes, feature Featurer, label Labeler) float64 {
	smooth := 1.0
	name := feature.Name()
	value := feature.Value()
	countFeature := nb.FeatureFreq[name][value][label]

	countRest := (nb.FeatureTotal[name][value] - countFeature)

	// crude smoothing
	if countFeature == 0 {
		countFeature = smooth
	}

	pFeature := countFeature / nb.LabelFreq[label]

	if countRest == 0 {
		countRest = smooth
	}
	// end crude smoothing

	pRest := countRest / (nb.Total - nb.LabelFreq[label])
	return pFeature / pRest
}
