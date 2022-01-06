package bayes

import (
	"errors"
	"fmt"

	ft "github.com/gnames/bayes/ent/feature"
	pst "github.com/gnames/bayes/ent/posterior"
)

type Option func(nb *bayes)

// OptPriorOdds allows dynamical change of prior odds used in calculations.
// Sometimes prior odds during classification event are very different from
// ones aquired during training. If for example 'real' prior odds are 100 times
// larger it means the calculated posterior odds will be 100 times smaller than
// what they would suppose to be.
func OptPriorOdds(lc map[ft.Class]int) Option {
	return func(nb *bayes) {
		nb.tmpClassCases = lc
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
	nb.tmpClassCases = nil
	nb.tmpCasesTotal = 0
	nb.ignorePriorOdds = false

	lc := nb.classCases
	ct := nb.casesTotal

	for _, opt := range opts {
		opt(nb)
	}

	if nb.tmpClassCases != nil {
		lc = nb.tmpClassCases
		ct = nb.tmpCasesTotal
	}

	l := len(lc)
	if l < 2 {
		return pst.Odds{}, errors.New("classes are empty")
	}
	return nb.multiPosterior(fs, lc, ct)
}

func (nb *bayes) noSuchFeature(f ft.Feature) bool {
	if _, ok := nb.featureCases[f]; ok {
		return false
	}
	return true
}

func (nb *bayes) multiPosterior(
	features []ft.Feature,
	classCases map[ft.Class]int,
	casesTotal int,
) (pst.Odds, error) {
	var maxClass ft.Class
	var maxOdds float64
	var res pst.Odds
	oddsPost := make(map[ft.Class]float64)
	likelihoods := make(pst.Likelihoods)

	for _, class := range nb.classes {
		odds, err := odds(class, classCases, casesTotal)
		if err != nil {
			return res, fmt.Errorf("cannot calculate odds: %s", err.Error())
		}
		oddsPost[class] = 1
		if !nb.ignorePriorOdds {
			oddsPost[class] = odds
		}
		likelihoods[class] = make(map[ft.Feature]float64)
		if !nb.ignorePriorOdds {
			po := ft.Feature{Name: "priorOdds", Value: "true"}
			likelihoods[class][po] = odds
		}

		var i int
		for _, f := range features {
			// features are missing if training data did not have
			// their value.
			if nb.noSuchFeature(f) {
				continue
			}

			lh, err := nb.Likelihood(f, class)
			if err != nil {
				return res, err
			}
			likelihoods[class][f] = lh

			i++
			oddsPost[class] *= lh
		}

		if i == 0 {
			return res, errors.New("all features are unknown")
		}

		if oddsPost[class] > maxOdds {
			maxOdds = oddsPost[class]
			maxClass = class
		}
	}
	p := pst.Odds{
		ClassOdds:   oddsPost,
		MaxClass:    maxClass,
		MaxOdds:     maxOdds,
		ClassCases:  classCases,
		Likelihoods: likelihoods,
	}
	return p, nil
}

func (nb *bayes) Likelihood(
	feature ft.Feature,
	class ft.Class,
) (float64, error) {
	err := nb.checkFeature(feature)
	if err != nil {
		return 0, err
	}
	err = nb.checkClass(class)
	if err != nil {
		return 0, err
	}
	smooth := 1

	countFeature := nb.featureCases[feature][class]

	countRest := (nb.featureTotal[feature] - countFeature)

	// crude smoothing to prevent fails for very unlikely cases.
	if countFeature == 0 {
		countFeature = smooth
	}

	if countRest == 0 {
		countRest = smooth
	}
	// end crude smoothing

	pFeature := float64(countFeature) / float64(nb.classCases[class])
	pRest := float64(countRest) / float64(nb.casesTotal-nb.classCases[class])
	return pFeature / pRest, nil
}
