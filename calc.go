package bayes

import (
	"errors"
	"fmt"

	ft "github.com/gnames/bayes/ent/feature"
	pst "github.com/gnames/bayes/ent/posterior"
)

type likelihoodsMap map[ft.Label]map[ft.Name]map[ft.Val]float64

type Option func(nb *bayes)

// OptPriorOdds allows dynamical change of prior odds used in calculations.
// Sometimes prior odds during classification event are very different from
// ones aquired during training. If for example 'real' prior odds are 100 times
// larger it means the calculated posterior odds will be 100 times smaller than
// what they would suppose to be.
func OptPriorOdds(lc map[ft.Label]float64) Option {
	return func(nb *bayes) {
		nb.tmpLabelCases = lc
		for _, v := range lc {
			nb.tmpCasesTotal += v
		}
	}
}

// OptIgnorePriorOdds might be needed if it is a muV
// PriorOdds already are accounted for.
func OptIgnorePriorOdds(b bool) Option {
	return func(nb *bayes) {
		nb.ignorePriorOdds = b
	}
}

// PosteriorOdds is a general function that runs NaiveBayes classifier against
// trained set. It can take a different PriorOdds value to influence
// calculation of the Posterior Odds.
func (nb *bayes) PosteriorOdds(
	fs []ft.Feature,
	opts ...Option,
) (pst.Odds, error) {
	nb.tmpLabelCases = nil
	nb.tmpCasesTotal = 0
	nb.ignorePriorOdds = false

	lc := nb.labelCases
	ct := nb.casesTotal

	for _, opt := range opts {
		opt(nb)
	}

	if nb.tmpLabelCases != nil {
		lc = nb.tmpLabelCases
		ct = nb.tmpCasesTotal
	}

	l := len(lc)
	if l < 2 {
		return pst.Odds{}, errors.New("labels are empty")
	}
	return nb.multiPosterior(fs, lc, ct)
}

func (nb *bayes) noSuchFeature(f ft.Feature) bool {
	name := f.Name
	if _, ok := nb.featureCases[name]; ok {
		value := f.Value
		if _, ok = nb.featureCases[name][value]; ok {
			return false
		}
	}
	return true
}

func (nb *bayes) multiPosterior(
	features []ft.Feature,
	labelCases map[ft.Label]float64,
	casesTotal float64,
) (pst.Odds, error) {
	var maxLabel ft.Label
	var maxOdds float64
	var res pst.Odds
	oddsPost := make(map[ft.Label]float64)
	likelihoods := make(likelihoodsMap)

	for _, label := range nb.labels {
		odds, err := odds(label, labelCases, casesTotal)
		if err != nil {
			return res, fmt.Errorf("cannot calculate odds: %s", err.Error())
		}
		oddsPost[label] = 1
		if !nb.ignorePriorOdds {
			oddsPost[label] = odds
		}
		likelihoods[label] = make(map[ft.Name]map[ft.Val]float64)
		if !nb.ignorePriorOdds {
			likelihoods[label][ft.Name("priorOdds")] =
				map[ft.Val]float64{ft.Val("true"): odds}
		}

		var i int
		for _, f := range features {
			// features are missing if training data did not have
			// their value.
			if nb.noSuchFeature(f) {
				continue
			}

			lh, _ := nb.Likelihood(f, label)
			likelihoods[label][f.Name] = map[ft.Val]float64{f.Value: lh}

			i++
			oddsPost[label] *= lh
		}

		if i == 0 {
			return res, errors.New("all features are unknown")
		}

		if oddsPost[label] > maxOdds {
			maxOdds = oddsPost[label]
			maxLabel = label
		}
	}
	p := pst.Odds{
		LabelOdds:   oddsPost,
		MaxLabel:    maxLabel,
		MaxOdds:     maxOdds,
		LabelCases:  labelCases,
		Likelihoods: likelihoods,
	}
	return p, nil
}

func (nb *bayes) Likelihood(
	feature ft.Feature,
	label ft.Label,
) (float64, error) {
	err := nb.checkFeature(feature)
	if err != nil {
		return 0, err
	}
	err = nb.checkLabel(label)
	if err != nil {
		return 0, err
	}
	smooth := 1.0
	name := feature.Name
	value := feature.Value

	countFeature := nb.featureCases[name][value][label]

	countRest := (nb.featureTotal[name][value] - countFeature)

	// crude smoothing to prevent fails for very unlikely cases.
	if countFeature == 0 {
		countFeature = smooth
	}

	if countRest == 0 {
		countRest = smooth
	}
	// end crude smoothing

	pFeature := countFeature / nb.labelCases[label]
	pRest := countRest / (nb.casesTotal - nb.labelCases[label])
	return pFeature / pRest, nil
}
