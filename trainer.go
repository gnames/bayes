package bayes

// TrainNB takes data from a training dataset and returns back a trained
// classifier.
func TrainNB(lfs []LabeledFeatures, opts ...OptionNB) *NaiveBayes {
	nb := NewNaiveBayes()
	for _, lf := range lfs {
		nb.LabelFreq[lf.Label]++
		trainFeatures(nb, lf)
	}

	for k, v := range nb.LabelFreq {
		nb.Labels = append(nb.Labels, k)
		nb.Total += v
	}

	featureTotal(nb)

	return nb
}

func trainFeatures(nb *NaiveBayes, lf LabeledFeatures) {
	for _, v := range lf.Features {
		name := v.Name()
		value := v.Value()
		if _, ok := nb.FeatureFreq[name]; !ok {
			nb.FeatureFreq[name] = make(map[FeatureValue]map[Labeler]float64)
		}
		if _, ok := nb.FeatureFreq[name][value]; !ok {
			nb.FeatureFreq[name][value] = make(map[Labeler]float64)
		}
		nb.FeatureFreq[name][value][lf.Label]++
	}
}

func featureTotal(nb *NaiveBayes) {
	for f, val := range nb.FeatureFreq {
		for fv, ls := range val {
			for _, v := range ls {
				if _, ok := nb.FeatureTotal[f]; !ok {
					nb.FeatureTotal[f] = make(map[FeatureValue]float64)
				}
				nb.FeatureTotal[f][fv] += v
			}
		}
	}
}
