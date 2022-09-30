// package output contains helpers to make results of odds calculation
// JSON friendly.
package output

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/gnames/bayes/ent/posterior"
)

type OddsDetails map[string]float64

func New(odds posterior.Odds, cl string) OddsDetails {
	res := make(OddsDetails)
	for class, fval := range odds.Likelihoods {
		if string(class) != cl {
			continue
		}
		for k, v := range fval {
			str := fmt.Sprintf("%s: %s", k.Name, k.Value)
			res[str] = v
		}
	}
	return res
}

type oddsOutput struct {
	Feature string  `json:"feature"`
	Odds    float64 `json:"odds"`
}

func (od OddsDetails) MarshalJSON() ([]byte, error) {
	odds := make([]oddsOutput, len(od))

	var i int
	for k, v := range od {
		odds[i] = oddsOutput{k, v}
		i++
	}

	sort.Slice(odds, func(i, j int) bool {
		return odds[i].Odds > odds[j].Odds
	})

	return json.Marshal(odds)
}

func (od OddsDetails) UnmarshalJSON(data []byte) error {
	var err error
	var odds []oddsOutput
	od = OddsDetails{}

	if err = json.Unmarshal(data, &odds); err != nil {
		return err
	}

	for _, v := range odds {
		od[v.Feature] = v.Odds
	}

	return nil
}
