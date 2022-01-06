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
	assert.Equal(t, 2, len(o.Classes))

	plain := o.FeatureCases["CookieF"]["plain"]
	assert.Equal(t, 30, plain["Jar1"])

	chocolate := o.FeatureCases["CookieF"]["chocolate"]
	assert.Equal(t, 15, chocolate["Jar2"])
	assert.Equal(t, 40, o.ClassCases["Jar1"])
}

// TestDumpLoad dumps the content of NaiveBayes object to json file
func TestDumpLoad(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)
	json, err := nb.Dump()
	assert.Nil(t, err)
	assert.Equal(t, "{\n  \"classes\": ", string(json)[0:15])
	nb2 := bayes.New()
	err = nb2.Load(json)
	assert.Nil(t, err)
	o := nb2.Inspect()
	assert.Equal(t, 70, o.CasesTotal)
	assert.Equal(t, 2, len(o.Classes))
	assert.Equal(t, 40, o.ClassCases["Jar1"])
	assert.Equal(t, 10, o.FeatureCases["CookieF"]["chocolate"]["Jar1"])
	assert.Equal(t, 30, o.FeatureCases["CookieF"]["plain"]["Jar1"])
}

func TestPriorOdds(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)

	t.Run("return prior odds", func(t *testing.T) {
		odds, err := nb.PriorOdds(ft.Class("Jar1"))
		assert.Nil(t, err)
		assert.InDelta(t, 1.333, odds, 0.01)
	})

	t.Run("returns error on unknown classes", func(t *testing.T) {
		_, err := nb.PriorOdds(ft.Class("Helicopter"))
		assert.Equal(t, "unkown class 'Helicopter'", err.Error())
	})

	t.Run("returns error if prior odds are infinite", func(t *testing.T) {
		nb := bayes.New()
		l := ft.Class("Surething")
		lf := ft.ClassFeatures{
			Class: l,
			Features: []ft.Feature{
				{Name: ft.Name("question"), Value: ft.Value("yes")},
			},
		}
		nb.Train([]ft.ClassFeatures{lf})
		o := nb.Inspect()
		assert.Equal(t, 1, o.CasesTotal)
		_, err := nb.PriorOdds(l)
		assert.Equal(t, "infinite prior odds", err.Error())
	})
}

func TestLikelihood(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)

	t.Run("calculates likelihood of one feature", func(t *testing.T) {
		f := ft.Feature{Name: "CookieF", Value: "plain"}

		odds, err := nb.Likelihood(f, ft.Class("Jar1"))
		assert.Nil(t, err)
		assert.Equal(t, odds, 1.5)
	})

	t.Run("returns error if feature name is not known", func(t *testing.T) {
		f := ft.Feature{Name: "donat", Value: "plain"}

		_, err := nb.Likelihood(f, ft.Class("Jar1"))
		assert.EqualError(t, err, "no feature with name 'donat' and value 'plain'")
	})

	t.Run("returns error if feature value is not known", func(t *testing.T) {
		f := ft.Feature{Name: "CookieF", Value: "wood"}

		_, err := nb.Likelihood(f, ft.Class("Jar1"))
		assert.EqualError(t, err, "no feature with name 'CookieF' and value 'wood'")
	})

	t.Run("returns error if label is not known", func(t *testing.T) {
		f := ft.Feature{Name: "CookieF", Value: "plain"}

		_, err := nb.Likelihood(f, ft.Class("Box"))
		assert.EqualError(t, err, "there is no label 'Box'")
	})
}

func TestPredict2Classes(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)

	t.Run("calculates posterior Probabilities", func(t *testing.T) {
		p, err := nb.PosteriorOdds([]ft.Feature{
			{Name: ft.Name("CookieF"), Value: ft.Value("plain")},
		})
		assert.Nil(t, err)
		assert.Equal(t, 2.0, p.MaxOdds)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
	})

	t.Run("calculates with new odds", func(t *testing.T) {
		lc := map[ft.Class]int{
			ft.Class("Jar1"): 1,
			ft.Class("Jar2"): 6,
		}
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("plain"),
				},
			},
			bayes.OptPriorOdds(lc),
		)
		assert.Nil(t, err)
		assert.InDelta(t, p.MaxOdds, 3.999, 0.1)
		assert.Equal(t, ft.Class("Jar2"), p.MaxClass)
	})

	t.Run("calculates without prior odds", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("plain"),
				},
			},
			bayes.OptIgnorePriorOdds(true),
		)
		assert.Nil(t, err)
		assert.Equal(t, 1.5, p.MaxOdds)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
		p, err = nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("plain"),
				},
			},
		)
		assert.Nil(t, err)
		assert.Equal(t, 2.0, p.MaxOdds)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
	})

	t.Run("calculates multiple posterior Probabilities", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("plain"),
				},
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("plain"),
				},
			},
		)
		assert.Nil(t, err)
		assert.Equal(t, 3.0, p.MaxOdds)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
	})

	t.Run("compensates infinite odds with crude smoothing", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name:  ft.Name("CookieF"),
				Value: ft.Value("chocolate"),
			},
			{
				Name:  ft.Name("ShapeF"),
				Value: ft.Value("star"),
			},
		}

		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
		assert.InDelta(t, 20, p.MaxOdds, 1)
	})

	t.Run("ignores features that are not in training", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name:  ft.Name("CookieF"),
				Value: ft.Value("plain"),
			},
			{
				Name:  ft.Name("ShapeF"),
				Value: ft.Value("square"),
			},
		}

		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
		assert.Equal(t, 2.0, p.MaxOdds)
	})

	t.Run("breaks on unknown features", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name: ft.Name("UnknownF"),
			},
		}
		_, err := nb.PosteriorOdds(f)
		assert.Equal(t, "all features are unknown", err.Error())
	})

	t.Run("ignores uknown features", func(t *testing.T) {
		f := []ft.Feature{
			{
				Name:  ft.Name("CookieF"),
				Value: ft.Value("plain"),
			},
			{
				Name: ft.Name("UnknownF"),
			},
		}
		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, 2.0, p.MaxOdds)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
	})
}

func TestPredict3Classes(t *testing.T) {
	lfs := threeCookieJarsFeatures()
	nb := bayes.New()
	nb.Train(lfs)

	t.Run("calculcates with 3 classes", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("chocolate"),
				},
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("chocolate"),
				},
			},
		)
		if err != nil {
			panic(err)
		}
		assert.Equal(t, ft.Class("Jar3"), p.MaxClass)
		assert.InDelta(t, 4.479, p.MaxOdds, 0.001)
	})

	t.Run("can calculate for 0 frequency", func(t *testing.T) {
		p, err := nb.PosteriorOdds(
			[]ft.Feature{
				{
					Name:  ft.Name("CookieF"),
					Value: ft.Value("plain"),
				},
			},
		)
		assert.Nil(t, err)
		assert.Equal(t, ft.Class("Jar1"), p.MaxClass)
		assert.InDelta(t, 2.0, p.MaxOdds, 0.01)
	})
}

// cookieJarsFeatures implements features from
// https://en.wikipedia.org/wiki/Bayesian_inference
// We change number of cookies in the second jar to 30, so
// prior odds are not 1:1.
// there are 2 jars with cookies. Jar1 has 10 chocolate and 30 vanilla cookies,
// jar2 has 15 of each. If a cookie is randomly taken from a random jar, what
// is a probability it came from the jar1?
func cookieJarsFeatures() []ft.ClassFeatures {
	n1 := ft.Name("CookieF")
	n2 := ft.Name("ShapeF")
	var lfs []ft.ClassFeatures

	for i := 1; i <= 40; i++ {
		v1 := ft.Value("chocolate")
		if i > 10 {
			v1 = ft.Value("plain")
		}
		f1 := ft.Feature{Name: n1, Value: v1}

		f2 := ft.Feature{Name: n2, Value: ft.Value("star")}
		lf := ft.ClassFeatures{
			Class:    ft.Class("Jar1"),
			Features: []ft.Feature{f1, f2},
		}
		lfs = append(lfs, lf)
	}

	for i := 1; i <= 30; i++ {
		v1 := ft.Value("chocolate")
		if i > 15 {
			v1 = ft.Value("plain")
		}
		f1 := ft.Feature{Name: n1, Value: v1}

		f2 := ft.Feature{Name: n2, Value: ft.Value("round")}
		lf := ft.ClassFeatures{
			Class:    ft.Class("Jar2"),
			Features: []ft.Feature{f1, f2},
		}
		lfs = append(lfs, lf)
	}
	return lfs
}

func threeCookieJarsFeatures() []ft.ClassFeatures {
	var f ft.Feature
	lfs := cookieJarsFeatures()
	for i := 1; i <= 40; i++ {
		f = ft.Feature{
			Name:  ft.Name("CookieF"),
			Value: ft.Value("chocolate"),
		}
		lf := ft.ClassFeatures{
			Class:    ft.Class("Jar3"),
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
	jar1 := ft.Class("Jar1")
	jar2 := ft.Class("Jar2")

	// Every preclassified feature-set provides data for one cookie. It tells
	// what jar has the cookie, what its kind and shape.
	cookie1 := ft.ClassFeatures{
		Class: jar1,
		Features: []ft.Feature{
			{Name: "kind", Value: "plain"},
			{Name: "shape", Value: "round"},
		},
	}
	cookie2 := ft.ClassFeatures{
		Class: jar1,
		Features: []ft.Feature{
			{Name: "kind", Value: "plain"},
			{Name: "shape", Value: "star"},
		},
	}
	cookie3 := ft.ClassFeatures{
		Class: jar1,
		Features: []ft.Feature{
			{Name: "kind", Value: "chocolate"},
			{Name: "shape", Value: "star"},
		},
	}
	cookie4 := ft.ClassFeatures{
		Class: jar1,
		Features: []ft.Feature{
			{Name: "kind", Value: "plain"},
			{Name: "shape", Value: "round"},
		},
	}
	cookie5 := ft.ClassFeatures{
		Class: jar1,
		Features: []ft.Feature{
			{Name: "kind", Value: "plain"},
			{Name: "shape", Value: "round"},
		},
	}
	cookie6 := ft.ClassFeatures{
		Class: jar2,
		Features: []ft.Feature{
			{Name: "kind", Value: "chocolate"},
			{Name: "shape", Value: "star"},
		},
	}
	cookie7 := ft.ClassFeatures{
		Class: jar2,
		Features: []ft.Feature{
			{Name: "kind", Value: "chocolate"},
			{Name: "shape", Value: "star"},
		},
	}
	cookie8 := ft.ClassFeatures{
		Class: jar2,
		Features: []ft.Feature{
			{Name: "kind", Value: "chocolate"},
			{Name: "shape", Value: "star"},
		},
	}

	lfs := []ft.ClassFeatures{
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
		{Name: ft.Name("kind"), Value: ft.Value("chocolate")},
		{Name: ft.Name("shape"), Value: ft.Value("star")},
	}

	res, err := nb.PosteriorOdds(aCookie)
	if err != nil {
		fmt.Println(err)
	}

	// it is more likely to that a random cookie comes from Jar1, but
	// for chocolate and star-shaped cookie it is more likely to come from
	// Jar2.
	fmt.Printf("Prior odds for Jar1 are %0.2f\n", oddsPrior)
	fmt.Printf("The cookie came from %s, with odds %0.2f\n", res.MaxClass, res.MaxOdds)
	// Output:
	// Prior odds for Jar1 are 1.67
	// The cookie came from Jar2, with odds 7.50
}
