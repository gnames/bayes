package bayes

import (
	"fmt"
)

// WithLaplaceSmoothing is an option that sets `true` value for
// LaplasSmoothing field in NaiveBayes classifier during training.
// This option is only needed if some features during training might
// generate 0.0 or 1.0 probabilities. Such probabilities need to be set
// to `0 < P < 1`. Smoothing skews resulting likelihoods, so it should
// be used only if it is necessary.
func WithLaplaceSmoothing(nb *NaiveBayes) error {
	nb.LaplaceSmoothing = true
	return nil
}

// WithLidstoneSmoothing is an option that assigns a float value for
// LaplasSmoothing field in NaiveBayes classifier during training.
// This option is only needed if some features during training might
// generate 0.0 or 1.0 probabilities. Often such probabilities need to be
// set to `0 < P < 1`. Smoothing skews resulting likelihoods, so it should
// be used only if it is necessary. Lidstone Smoothing coefficient should
// not be 0 < a < 1.
func WithLidstoneSmoothing(a float64) func(*NaiveBayes) error {
	return func(nb *NaiveBayes) error {
		if a <= 0 || a >= 1 {
			return fmt.Errorf("Coefficient %0.2f is out of range", a)
		}
		nb.LidstoneSmoothing = a
		return nil
	}
}

// WithLanguage is an option that assignes a language to a training set.
func WithLanguage(l string) func(*NaiveBayes) error {
	return func(nb *NaiveBayes) error {
		if l != "en" {
			return fmt.Errorf("Unknown language %s", l)
		}
		nb.Language = l
		return nil
	}
}

// TrainNB takes data from a training dataset and returns back a trained
// classifier.
func TrainNB(lfs []LabeledFeatures, opts ...OptionNB) *NaiveBayes {
	nb := NewNaiveBayes(opts...)
	for _, lf := range lfs {
		trainFeatures(nb, lf)
	}

	for k := range nb.LabelFreq {
		nb.Labels = append(nb.Labels, k)
	}

	if nb.LidstoneSmoothing > 0.0 {
		nb.AddLidstoneSmoothing()
	} else if nb.LaplaceSmoothing {
		nb.AddLaplaceSmoothing()
	}

	featureTotal(nb)

	return nb
}

// AddLaplaceSmoothing is introduced for features that reach 1.0 or 0 in their
// probabilities for a hypothesis. For example
//
//   P(penguin|noFly) = 1
//
// and if we try to find corresponding likelihood
//
//   likelihood = P(penguin|noFly)/P(penguin|Fly)
//
// it is going to be equal +Infinity. If we add 'fake' data points that assume
// we did meet a flying penguin just once, and for compensation we also add
// one more non-flying penguin, we introduce Laplace smoothing. Now all
// probabilities are larger than 0 and smaller than 1.
func (nb *NaiveBayes) AddLaplaceSmoothing() {
	for f := range nb.FeatureFreq {
		for _, l := range nb.Labels {
			nb.FeatureFreq[f][l]++
			nb.LabelFreq[l]++
		}
	}
}

// AddLidstoneSmoothing is similar to AddLaplaceSmoothing, but the smoothing
// coefficient is a float64 value between 0 and 1. As a result it creates
// less skewing of the outcomes than Laplace smoothing.
func (nb *NaiveBayes) AddLidstoneSmoothing() {
	for f := range nb.FeatureFreq {
		for _, l := range nb.Labels {
			nb.FeatureFreq[f][l] += nb.LidstoneSmoothing
			nb.LabelFreq[l] += nb.LidstoneSmoothing
		}
	}
}

func trainFeatures(nb *NaiveBayes, lf LabeledFeatures) {
	for _, v := range lf.Features {
		name := v.Name()
		if _, ok := nb.FeatureFreq[name]; !ok {
			nb.FeatureFreq[name] = make(map[Label]float64)
		}
		nb.FeatureFreq[name][lf.Label]++
		nb.LabelFreq[lf.Label]++
	}
}

func featureTotal(nb *NaiveBayes) {
	for feature, labels := range nb.FeatureFreq {
		for _, v := range labels {
			nb.FeatureTotal[feature] += v
			nb.Total += v
		}
	}
}
