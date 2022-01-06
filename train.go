package bayes

import (
	ft "github.com/gnames/bayes/ent/feature"
)

func (nb *bayes) Train(lfs []ft.ClassFeatures) {
	for i := range lfs {
		nb.classCases[lfs[i].Class]++
		nb.trainFeatures(lfs[i])
	}

	nb.classes = make([]ft.Class, len(nb.classCases))
	var count int
	for k, v := range nb.classCases {
		nb.classes[count] = k
		nb.casesTotal += v
		count++
	}
	nb.featTotal()
}

func (nb *bayes) trainFeatures(lf ft.ClassFeatures) {
	for _, v := range lf.Features {
		if _, ok := nb.featureCases[v]; !ok {
			nb.featureCases[v] = make(map[ft.Class]int)
		}
		nb.featureCases[v][lf.Class]++
	}
}
