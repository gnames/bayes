package bayes

import (
	ft "github.com/gnames/bayes/ent/feature"
)

func (nb *bayes) Train(lfs []ft.LabeledFeatures) {
	for i := range lfs {
		nb.labelCases[lfs[i].Label]++
		nb.trainFeatures(lfs[i])
	}

	nb.labels = make([]ft.Label, len(nb.labelCases))
	var count int
	for k, v := range nb.labelCases {
		nb.labels[count] = k
		nb.casesTotal += v
		count++
	}
	nb.featTotal()
}

func (nb *bayes) trainFeatures(lf ft.LabeledFeatures) {
	for _, v := range lf.Features {
		if _, ok := nb.featureCases[v.Name]; !ok {
			nb.featureCases[v.Name] = make(map[ft.Val]map[ft.Label]float64)
		}
		if _, ok := nb.featureCases[v.Name][v.Value]; !ok {
			nb.featureCases[v.Name][v.Value] = make(map[ft.Label]float64)
		}
		nb.featureCases[v.Name][v.Value][lf.Label]++
	}
}
