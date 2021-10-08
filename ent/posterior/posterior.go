package posterior

import ft "github.com/gnames/bayes/ent/feature"

type Odds struct {
	LabelOdds   map[ft.Label]float64
	MaxLabel    ft.Label
	MaxOdds     float64
	LabelFreq   map[ft.Label]float64
	Likelihoods map[ft.Label]map[ft.Name]map[ft.Val]float64
}
