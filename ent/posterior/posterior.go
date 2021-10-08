package posterior

import ft "github.com/gnames/bayes/ent/feature"

type Odds struct {
	LabelOdds   map[ft.Label]float64                        `json:"labelOdds"`
	MaxLabel    ft.Label                                    `json:"maxLabel"`
	MaxOdds     float64                                     `json:"maxOdds"`
	LabelCases  map[ft.Label]float64                        `json:"labelCases"`
	Likelihoods map[ft.Label]map[ft.Name]map[ft.Val]float64 `json:"likelihoods"`
}
