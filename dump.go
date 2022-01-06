package bayes

import (
	"bytes"
	"encoding/json"

	"github.com/gnames/bayes/ent/bayesdump"
	ft "github.com/gnames/bayes/ent/feature"
)

// Dump serializes a Bayes object into a JSON format.
func (nb *bayes) Dump() ([]byte, error) {
	return json.MarshalIndent(nb, "", "  ")
}

// Load deserializes a JSON text into Bayes object. The function needs
// to know how to convert a string that represents a class to an object.
func (nb *bayes) Load(dump []byte) error {
	r := bytes.NewReader(dump)
	return json.NewDecoder(r).Decode(nb)
}

// MarshalJSON serializes a NaiveBayes object to JSON.
func (nb *bayes) MarshalJSON() ([]byte, error) {
	res := nb.Inspect()
	return json.Marshal(&res)
}

func (nb *bayes) Inspect() bayesdump.BayesDump {
	ls := make([]string, len(nb.classes))
	for i, v := range nb.classes {
		ls[i] = string(v)
	}

	lfs := make(map[string]int)
	for k, v := range nb.classCases {
		lfs[string(k)] = v
	}

	ffs := make(map[string]map[string]map[string]int)
	for fk, fv := range nb.featureCases {
		if _, ok := ffs[string(fk.Name)]; !ok {
			val1 := make(map[string]map[string]int)
			ffs[string(fk.Name)] = val1
		}
		if _, ok := ffs[string(fk.Name)][string(fk.Value)]; !ok {
			val2 := make(map[string]int)
			ffs[string(fk.Name)][string(fk.Value)] = val2
		}
		for lk, v := range fv {
			ffs[string(fk.Name)][string(fk.Value)][string(lk)] = v
		}
	}
	return bayesdump.BayesDump{
		Classes:      ls,
		CasesTotal:   nb.casesTotal,
		ClassCases:   lfs,
		FeatureCases: ffs,
	}
}

// UnmarshalJSON deserializes JSON data to a NaiveBayes object.
func (nb *bayes) UnmarshalJSON(data []byte) error {
	var res bayesdump.BayesDump
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}

	nb.classes = make([]ft.Class, len(res.Classes))
	for i, v := range res.Classes {
		nb.classes[i] = ft.Class(v)
	}

	nb.casesTotal = res.CasesTotal

	for k, v := range res.ClassCases {
		nb.classCases[ft.Class(k)] = v
	}

	for k1, v1 := range res.FeatureCases {
		name := ft.Name(k1)
		for k2, v2 := range v1 {
			v := make(map[ft.Class]int)
			f := ft.Feature{Name: name, Value: ft.Value(k2)}
			nb.featureCases[f] = v
			for k3, v3 := range v2 {
				class := ft.Class(k3)
				nb.featureCases[f][class] = v3
			}
		}
	}

	nb.featTotal()
	return nil
}
