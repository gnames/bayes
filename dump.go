package bayes

import (
	"bytes"
	"encoding/json"
)

// Dump serializes a NaiveBayes object into a JSON format.
func (nb *NaiveBayes) Dump() []byte {
	// json, err := jsoniter.MarshalIndent(nb, "", "  ")
	json, err := json.Marshal(nb)
	if err != nil {
		panic(err)
	}
	return json
}

// Restore deserializes a JSON text into NaiveBayes object. The function needs
// to know how to convert a string that represents a label to an object. Use
// RegisterLabel function to inject a string-to-Label conversion map.
func (nb *NaiveBayes) Restore(dump []byte) {
	r := bytes.NewReader(dump)
	err := json.NewDecoder(r).Decode(nb)
	if err != nil {
		panic(err)
	}
}

type nbTemp NaiveBayes
type featureFreqJSON map[FeatureName]map[FeatureValue]map[string]float64
type nbJSON struct {
	Labels      []string           `json:"labels"`
	LabelFreq   map[string]float64 `json:"label_freq"`
	FeatureFreq featureFreqJSON    `json:"feature_freq"`
	*nbTemp
}

// MarshalJSON serializes a NaiveBayes object to JSON.
func (nb *NaiveBayes) MarshalJSON() ([]byte, error) {
	ls := make([]string, len(nb.Labels))
	for i, v := range nb.Labels {
		ls[i] = v.String()
	}
	lfs := make(map[string]float64)
	for k, v := range nb.LabelFreq {
		lfs[k.String()] = v
	}
	ffs := make(map[FeatureName]map[FeatureValue]map[string]float64)
	for k1, v1 := range nb.FeatureFreq {
		val := make(map[FeatureValue]map[string]float64)
		ffs[k1] = val
		for k2, v2 := range v1 {
			val := make(map[string]float64)
			ffs[k1][k2] = val
			for k3, v3 := range v2 {
				ffs[k1][k2][k3.String()] = v3
			}
		}
	}
	res := nbJSON{
		Labels:      ls,
		LabelFreq:   lfs,
		FeatureFreq: ffs,
		nbTemp:      (*nbTemp)(nb),
	}
	return json.MarshalIndent(&res, "", "  ")
}

// UnmarshalJSON deserializes JSON data to a NaiveBayes object.
func (nb *NaiveBayes) UnmarshalJSON(data []byte) (err error) {
	var l Labeler
	res := nbJSON{nbTemp: (*nbTemp)(nb)}
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}
	ls := make([]Labeler, len(res.Labels))
	for i, v := range res.Labels {
		if l, err = LabelFactory(v); err != nil {
			return err
		}
		ls[i] = l
	}
	nb.Labels = ls
	lfs := make(map[Labeler]float64)
	for k, v := range res.LabelFreq {
		if l, err = LabelFactory(k); err != nil {
			return err
		}
		lfs[l] = v
	}
	nb.LabelFreq = lfs
	for k1, v1 := range res.FeatureFreq {
		val := make(map[FeatureValue]map[Labeler]float64)
		nb.FeatureFreq[k1] = val
		for k2, v2 := range v1 {
			val := make(map[Labeler]float64)
			nb.FeatureFreq[k1][k2] = val
			for k3, v3 := range v2 {
				if l, err = LabelFactory(k3); err != nil {
					return err
				}
				nb.FeatureFreq[k1][k2][l] = v3
			}
		}
	}
	return nil
}
