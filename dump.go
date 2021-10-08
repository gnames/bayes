package bayes

import (
	"bytes"
	"encoding/json"

	ft "github.com/gnames/bayes/ent/feature"
	"github.com/gnames/bayes/ent/output"
)

// Dump serializes a Bayes object into a JSON format.
func (nb *bayes) Dump() ([]byte, error) {
	return json.MarshalIndent(nb, "", "  ")
}

// Load deserializes a JSON text into Bayes object. The function needs
// to know how to convert a string that represents a label to an object.
func (nb *bayes) Load(dump []byte) error {
	r := bytes.NewReader(dump)
	return json.NewDecoder(r).Decode(nb)
}

// MarshalJSON serializes a NaiveBayes object to JSON.
func (nb *bayes) MarshalJSON() ([]byte, error) {
	res := nb.Inspect()
	return json.MarshalIndent(&res, "", "  ")
}

func (nb *bayes) Inspect() output.Output {
	ls := make([]string, len(nb.labels))
	for i, v := range nb.labels {
		ls[i] = string(v)
	}

	lfs := make(map[string]float64)
	for k, v := range nb.labelCases {
		lfs[string(k)] = v
	}

	ffs := make(map[string]map[string]map[string]float64)
	for k1, v1 := range nb.featureCases {
		val := make(map[string]map[string]float64)
		ffs[string(k1)] = val
		for k2, v2 := range v1 {
			val := make(map[string]float64)
			ffs[string(k1)][string(k2)] = val
			for k3, v3 := range v2 {
				ffs[string(k1)][string(k2)][string(k3)] = v3
			}
		}
	}
	return output.Output{
		Labels:       ls,
		CasesTotal:   nb.casesTotal,
		LabelCases:   lfs,
		FeatureCases: ffs,
	}
}

// UnmarshalJSON deserializes JSON data to a NaiveBayes object.
func (nb *bayes) UnmarshalJSON(data []byte) error {
	var res output.Output
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}

	nb.labels = make([]ft.Label, len(res.Labels))
	for i, v := range res.Labels {
		nb.labels[i] = ft.Label(v)
	}

	nb.casesTotal = res.CasesTotal

	for k, v := range res.LabelCases {
		nb.labelCases[ft.Label(k)] = v
	}

	for k1, v1 := range res.FeatureCases {
		val := make(map[ft.Val]map[ft.Label]float64)
		name := ft.Name(k1)
		nb.featureCases[name] = val
		for k2, v2 := range v1 {
			v := make(map[ft.Label]float64)
			value := ft.Val(k2)
			nb.featureCases[name][value] = v
			for k3, v3 := range v2 {
				label := ft.Label(k3)
				nb.featureCases[name][value][label] = v3
			}
		}
	}

	nb.featTotal()
	return nil
}
