package bayes

import (
	"errors"
	"fmt"

	ft "github.com/gnames/bayes/ent/feature"
)

type bayes struct {
	// classes are a classifier categories. There must be at least 2 classes.
	classes []ft.Class

	// casesTotal is the number of entries in the training set including
	// all classes.
	casesTotal int

	// classCases is the number of entities per one class.
	classCases map[ft.Class]int

	// featureCases is the number of entities per a particular feature.
	featureCases map[ft.Feature]map[ft.Class]int

	// featureTotal is the number of all entities for a perticular feature.
	featureTotal map[ft.Feature]int

	// tmpClassCases is used to provide a new prior odds.
	tmpClassCases map[ft.Class]int

	// tmpCasesTotal is used to provide a new prior odds.
	tmpCasesTotal int

	// ignorePriorOdds indicates that likelihood will be calculated without
	// taking in account prior odds.
	ignorePriorOdds bool
}

// New creates a new instance of Bayes object. This object needs to get data
// from either training or from loading a dump of previous training data.
func New() Bayes {
	nb := &bayes{
		classCases:   make(map[ft.Class]int),
		featureCases: make(map[ft.Feature]map[ft.Class]int),
		featureTotal: make(map[ft.Feature]int),
	}
	return nb
}

// PriorOdds returns prior odds calculated from the training set
func (nb *bayes) PriorOdds(l ft.Class) (float64, error) {
	return odds(l, nb.classCases, nb.casesTotal)
}

func odds(
	l ft.Class,
	lc map[ft.Class]int,
	casesTotal int,
) (float64, error) {
	var freq int
	var ok bool
	if freq, ok = lc[l]; !ok {
		return 0, fmt.Errorf("unknown class '%s'", l)
	}
	pL := float64(freq) / float64(casesTotal)
	if pL == 1 || casesTotal == 0 {
		return 0, errors.New("infinite prior odds")
	}
	return pL / (1 - pL), nil
}

func (b *bayes) Odds(lfs ft.ClassFeatures) (float64, error) {
	return 0, nil
}

func (nb *bayes) checkFeature(f ft.Feature) error {
	var ok bool
	if _, ok = nb.featureCases[f]; !ok {
		return fmt.Errorf("no feature with name '%s' and value '%s'", f.Name, f.Value)
	}
	return nil
}

func (nb *bayes) checkClass(l ft.Class) error {
	if _, ok := nb.classCases[l]; !ok {
		return fmt.Errorf("there is no label '%s'", l)
	}
	return nil
}

func (nb *bayes) featTotal() {
	for fk, fv := range nb.featureCases {
		for _, v := range fv {
			nb.featureTotal[fk] += v
		}
	}
}
