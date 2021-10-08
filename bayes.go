package bayes

import (
	"errors"
	"fmt"

	ft "github.com/gnames/bayes/ent/feature"
)

type bayes struct {
	labels          []ft.Label
	casesTotal      float64
	labelCases      map[ft.Label]float64
	featureCases    map[ft.Name]map[ft.Val]map[ft.Label]float64
	featureTotal    map[ft.Name]map[ft.Val]float64
	tmpLabelCases   map[ft.Label]float64
	tmpCasesTotal   float64
	ignorePriorOdds bool
}

// New creates a new instance of Bayes object. This object needs to get data
// from either training or from loading a dump of previous training data.
func New() Bayes {
	nb := &bayes{
		labelCases:   make(map[ft.Label]float64),
		featureCases: make(map[ft.Name]map[ft.Val]map[ft.Label]float64),
		featureTotal: make(map[ft.Name]map[ft.Val]float64),
	}
	return nb
}

// PriorOdds returns prior odds calculated from the training set
func (nb *bayes) PriorOdds(l ft.Label) (float64, error) {
	return odds(l, nb.labelCases, nb.casesTotal)
}

func odds(
	l ft.Label,
	lc map[ft.Label]float64,
	casesTotal float64,
) (float64, error) {
	var freq float64
	var ok bool
	if freq, ok = lc[l]; !ok {
		return 0, fmt.Errorf("unkown label '%s'", l)
	}
	pL := freq / casesTotal
	if pL == 1 || casesTotal == 0 {
		return 0, errors.New("infinite prior odds")
	}
	return pL / (1 - pL), nil
}

func (b *bayes) Odds(lfs ft.LabeledFeatures) (float64, error) {
	return 0, nil
}

func (nb *bayes) featTotal() {
	for nk, nv := range nb.featureCases {
		for vk, vv := range nv {
			for _, lv := range vv {
				if _, ok := nb.featureTotal[nk]; !ok {
					nb.featureTotal[nk] = make(map[ft.Val]float64)
				}
				nb.featureTotal[nk][vk] += lv
			}
		}
	}
}
