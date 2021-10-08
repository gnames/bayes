package bayes_test

import (
	"fmt"
	"log"
	"testing"

	"github.com/gnames/bayes"
	ft "github.com/gnames/bayes/ent/feature"
	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	nb := bayes.New()
	_, ok := nb.(bayes.Trainer)
	assert.True(t, ok)
	_, ok = nb.(bayes.Serializer)
	assert.True(t, ok)
	_, ok = nb.(bayes.Calc)
	assert.True(t, ok)
}

func TestTrain(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)
	o := nb.Inspect()
	assert.Equal(t, len(o.Labels), 2)
	plain := o.FeatureCases["CookieF"]["plain"]
	assert.Equal(t, plain["Jar1"], 30.0)
	chocolate := o.FeatureCases["CookieF"]["chocolate"]
	assert.Equal(t, chocolate["Jar2"], 15.0)
	assert.Equal(t, o.LabelCases["Jar1"], 40.0)
}

// TestDumpLoad dumps the content of NaiveBayes object to json file
func TestDumpLoad(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)
	json, err := nb.Dump()
	assert.Nil(t, err)
	assert.Equal(t, string(json)[0:15], "{\n  \"labels\": [")
	nb2 := bayes.New()
	err = nb2.Load(json)
	assert.Nil(t, err)
	o := nb2.Inspect()
	assert.Equal(t, o.CasesTotal, 70.0)
	assert.Equal(t, len(o.Labels), 2)
	assert.Equal(t, o.LabelCases["Jar1"], 40.0)
	assert.Equal(t, o.FeatureCases["CookieF"]["chocolate"]["Jar1"], 10.0)
}

func TestPriorOdds(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)

	t.Run("return prior odds", func(t *testing.T) {
		odds, err := nb.PriorOdds(ft.Label("Jar1"))
		assert.Nil(t, err)
		assert.InDelta(t, odds, 1.333, 0.01)
	})

	t.Run("returns error on unkown labels", func(t *testing.T) {
		_, err := nb.PriorOdds(ft.Label("Helicopter"))
		assert.Equal(t, err.Error(), "unkown label 'Helicopter'")
	})

	t.Run("returns error if prior odds are infinite", func(t *testing.T) {
		nb := bayes.New()
		l := ft.Label("Surething")
		lf := ft.LabeledFeatures{
			Label: l,
			Features: []ft.Feature{
				{Name: ft.Name("question"), Value: ft.Val("yes")},
			},
		}
		nb.Train([]ft.LabeledFeatures{lf})
		o := nb.Inspect()
		assert.Equal(t, o.CasesTotal, 1.0)
		_, err := nb.PriorOdds(l)
		assert.Equal(t, err.Error(), "infinite prior odds")
	})
}

func TestPredict2Labels(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)

	t.Run("calculates posterior Probabilities", func(t *testing.T) {
		p, err := nb.PosteriorOdds([]ft.Feature{
			{Name: ft.Name("CookieF"), Value: ft.Val("plain")},
		})
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 2.0)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
	})

	t.Run("calculates with new odds", func(t *testing.T) {
		lc := map[ft.Label]float64{
			ft.Label("Jar1"): 1,
			ft.Label("Jar2"): 6,
		}
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("plain"),
				},
			},
			bayes.OptPriorOdds(lc),
		)
		assert.Nil(t, err)
		assert.InDelta(t, p.MaxOdds, 3.999, 0.1)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar2"))
	})

	t.Run("calculates without prior odds", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("plain"),
				},
			},
			bayes.OptIgnorePriorOdds(true),
		)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 1.5)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
		p, err = nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("plain"),
				},
			},
		)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 2.0)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
	})

	t.Run("calculates multiple posterior Probabilities", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("plain"),
				},
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("plain"),
				},
			},
		)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 3.0)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
	})

	t.Run("compensates infinite odds with crude smoothing", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name:  ft.Name("CookieF"),
				Value: ft.Val("chocolate"),
			},
			{
				Name:  ft.Name("ShapeF"),
				Value: ft.Val("star"),
			},
		}

		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
		assert.InDelta(t, p.MaxOdds, 20, 1)
	})

	t.Run("ignores features that are not in training", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name:  ft.Name("CookieF"),
				Value: ft.Val("plain"),
			},
			{
				Name:  ft.Name("ShapeF"),
				Value: ft.Val("square"),
			},
		}

		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
		assert.Equal(t, p.MaxOdds, 2.0)
	})

	t.Run("breaks on unknown features", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name: ft.Name("UnknownF"),
			},
		}
		_, err := nb.PosteriorOdds(f)
		assert.Equal(t, err.Error(), "all features are unknown")
	})

	t.Run("ignores uknown features", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name:  ft.Name("CookieF"),
				Value: ft.Val("plain"),
			},
			{
				Name: ft.Name("UnknownF"),
			},
		}
		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 2.0)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
	})
}

func TestPredict3Labels(t *testing.T) {
	lfs := threeCookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)

	t.Run("calculcates with 3 labels", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("chocolate"),
				},
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("chocolate"),
				},
			},
		)
		if err != nil {
			panic(err)
		}
		assert.Equal(t, p.MaxLabel, ft.Label("Jar3"))
		assert.InDelta(t, p.MaxOdds, 4.479, 0.001)
	})

	t.Run("can calculate for 0 frequency", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Val("plain"),
				},
			},
		)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxLabel, ft.Label("Jar1"))
		assert.InDelta(t, p.MaxOdds, 2.0, 0.01)
	})
}

// cookieJarsFeatures implements features from
// https://en.wikipedia.org/wiki/Bayesian_inference
// We change number of cookies in the second jar to 30, so
// prior odds are not 1:1.
// there are 2 jars with cookies. Jar1 has 10 chocolate and 30 vanilla cookies,
// jar2 has 15 of each. If a cookie is randomly taken from a random jar, what
// is a probability it came from the jar1?
func cookieJarsFeatures() []ft.LabeledFeatures {
	n1 := ft.Name("CookieF")
	n2 := ft.Name("ShapeF")
	var lfs []ft.LabeledFeatures

	for i := 1; i <= 40; i++ {
		v1 := ft.Val("chocolate")
		if i > 10 {
			v1 = ft.Val("plain")
		}
		f1 := ft.Feature{Name: n1, Value: v1}

		f2 := ft.Feature{Name: n2, Value: ft.Val("star")}
		lf := ft.LabeledFeatures{
			Label:    ft.Label("Jar1"),
			Features: []ft.Feature{f1, f2},
		}
		lfs = append(lfs, lf)
	}

	for i := 1; i <= 30; i++ {
		v1 := ft.Val("chocolate")
		if i > 15 {
			v1 = ft.Val("plain")
		}
		f1 := ft.Feature{Name: n1, Value: v1}

		f2 := ft.Feature{Name: n2, Value: ft.Val("round")}
		lf := ft.LabeledFeatures{
			Label:    ft.Label("Jar2"),
			Features: []ft.Feature{f1, f2},
		}
		lfs = append(lfs, lf)
	}
	return lfs
}

func threeCookieJarsFeatures() []ft.LabeledFeatures {
	var f ft.Feature
	lfs := cookieJarsFeatures()
	for i := 1; i <= 40; i++ {
		f = ft.Feature{
			Name:  ft.Name("CookieF"),
			Value: ft.Val("chocolate"),
		}
		lf := ft.LabeledFeatures{
			Label:    ft.Label("Jar3"),
			Features: []ft.Feature{f},
		}
		lfs = append(lfs, lf)
	}
	return lfs
}

func Example() {
	// there are two jars of cookies, they are our training set.
	// Cookies have be round or star-shaped.
	// There are plain or chocolate chips cookies.
	jar1 := ft.Label("Jar1")
	jar2 := ft.Label("Jar2")

	// Every labeled feature-set provides data for one cookie. It tells
	// what jar has the cookie, what its kind and shape.
	cookie1 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("round")},
		},
	}
	cookie2 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie3 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie4 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("round")},
		},
	}
	cookie5 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("round")},
		},
	}
	cookie6 := ft.LabeledFeatures{
		Label: jar2,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie7 := ft.LabeledFeatures{
		Label: jar2,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie8 := ft.LabeledFeatures{
		Label: jar2,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}

	lfs := []ft.LabeledFeatures{
		cookie1, cookie2, cookie3, cookie4, cookie5, cookie6, cookie7, cookie8,
	}

	nb := bayes.New()
	nb.Train(lfs)
	oddsPrior, err := nb.PriorOdds(jar1)
	if err != nil {
		log.Println(err)
	}

	// If we got a chocolate star-shaped cookie, which jar it came from most
	// likely?
	aCookie := []ft.Feature{
		{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
		{Name: ft.Name("shape"), Value: ft.Val("star")},
	}

	res, err := nb.PosteriorOdds(aCookie)
	if err != nil {
		fmt.Println(err)
	}

	// it is more likely to that a random cookie comes from Jar1, but
	// for chocolate and star-shaped cookie it is more likely to come from
	// Jar2.
	fmt.Printf("Prior odds for Jar1 are %0.2f\n", oddsPrior)
	fmt.Printf("The cookie came from %s, with odds %0.2f\n", res.MaxLabel, res.MaxOdds)
	// Output:
	// Prior odds for Jar1 are 1.67
	// The cookie came from Jar2, with odds 7.50
}
