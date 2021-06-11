package bayes_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/gnames/bayes"
	"github.com/stretchr/testify/assert"
)

// bayes.RegisterLabel(labelsDict)

func TestNew(t *testing.T) {
	nb := bayes.NewNaiveBayes()
	assert.Equal(t, len(nb.Labels), 0)
	assert.NotEqual(t, nb, bayes.NaiveBayes{})
}

func TestTrainNB(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.TrainNB(lfs)
	assert.Equal(t, len(nb.Labels), 2)
	plain := nb.FeatureFreq[bayes.FeatureName("CookieF")][bayes.FeatureValue("plain")]
	assert.Equal(t, plain[Jar1], 30.0)
	chocolate := nb.FeatureFreq[bayes.FeatureName("CookieF")][bayes.FeatureValue("chocolate")]
	assert.Equal(t, chocolate[Jar2], 15.0)
	assert.Equal(t, nb.LabelFreq[Jar1], 40.0)
}

// TestDumpRestore dumps the content of NaiveBayes object to json file
func TestDumpRestore(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.TrainNB(lfs)
	json := nb.Dump()
	assert.Equal(t, string(json)[0:15], "{\"labels\":[\"Jar")
	nb2 := bayes.NewNaiveBayes()
	nb2.Restore(json)
	assert.Equal(t, nb2.Total, 70.0)
	assert.Equal(t, len(nb2.Labels), 2)
	assert.Equal(t, nb2.LabelFreq[Jar1], 40.0)
	assert.Equal(t, nb2.
		FeatureFreq[bayes.FeatureName("CookieF")][bayes.FeatureValue("chocolate")][Jar1],
		10.0)
}

func TestTrainingPrior(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.TrainNB(lfs)

	t.Run("return prior odds", func(t *testing.T) {
		odds, err := nb.TrainingPrior(Jar1)
		assert.Nil(t, err)
		assert.InDelta(t, odds, 1.333, 0.01)
	})

	t.Run("returns error on unkown labels", func(t *testing.T) {
		_, err := nb.TrainingPrior(Helicopter)
		assert.Equal(t, err.Error(), "Unkown label \"Helicopter\"")
	})

	t.Run("returns error if prior odds are infinite", func(t *testing.T) {
		nb := bayes.NewNaiveBayes()
		l := Surething
		nb.LabelFreq[l] = 10.0
		nb.Total = 10.0
		_, err := nb.TrainingPrior(l)
		assert.Equal(t, err.Error(), "Infinite prior odds")
	})
}

func TestOdds(t *testing.T) {
	t.Run("returns odds from frequencies", func(t *testing.T) {
		lfs := cookieJarsFeatures()
		nb := bayes.TrainNB(lfs)
		odds, err := bayes.Odds(Jar1, nb.LabelFreq)
		assert.Nil(t, err)
		assert.InDelta(t, odds, 1.33, 0.01)
	})

	t.Run("calculates 'local' frequences correctly", func(t *testing.T) {
		l1 := Lots
		l2 := Little
		lf := make(bayes.LabelFreq)
		lf[l1] = 9.0
		lf[l2] = 1.0
		odds, err := bayes.Odds(Lots, lf)
		assert.Nil(t, err)
		assert.InDelta(t, odds, 9.00, 0.01)
	})

	t.Run("breaks if frequences are not sensical", func(t *testing.T) {
		l := Surething
		lf := make(bayes.LabelFreq)
		lf[l] = 10.0
		_, err := bayes.Odds(Surething, lf)
		assert.Equal(t, err.Error(), "Infinite prior odds")
	})
}

func TestPredict2Labels(t *testing.T) {
	lfs := cookieJarsFeatures()
	nb := bayes.TrainNB(lfs)

	t.Run("calculates posterior Probabilities", func(t *testing.T) {
		p, err := nb.PosteriorOdds([]bayes.Featurer{CookieF{"plain"}})
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 2.0)
		assert.Equal(t, p.MaxLabel, Jar1)
	})

	t.Run("calculates with new odds", func(t *testing.T) {
		lf := bayes.LabelFreq{
			Jar1: 1,
			Jar2: 6,
		}
		p, err := nb.PosteriorOdds([]bayes.Featurer{CookieF{"plain"}},
			bayes.WithPriorOdds(lf))
		assert.Nil(t, err)
		assert.InDelta(t, p.MaxOdds, 3.999, 0.1)
		assert.Equal(t, p.MaxLabel, Jar2)
	})

	t.Run("calculates without prior odds", func(t *testing.T) {
		p, err := nb.PosteriorOdds([]bayes.Featurer{CookieF{"plain"}},
			bayes.IgnorePriorOdds)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 1.5)
		assert.Equal(t, p.MaxLabel, Jar1)
		p, _ = nb.PosteriorOdds([]bayes.Featurer{CookieF{"plain"}})
		assert.Equal(t, p.MaxOdds, 2.0)
		assert.Equal(t, p.MaxLabel, Jar1)
	})

	t.Run("calculates multiple posterior Probabilities", func(t *testing.T) {
		p, err := nb.PosteriorOdds([]bayes.Featurer{CookieF{"plain"},
			CookieF{"plain"}})
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 3.0)
		assert.Equal(t, p.MaxLabel, Jar1)
	})

	t.Run("compensates infinite odds with crude smoothing", func(t *testing.T) {
		f := []bayes.Featurer{CookieF{"chocolate"}, ShapeF{"star"}}

		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxLabel, Jar1)
		assert.InDelta(t, p.MaxOdds, 20, 1)
	})

	t.Run("ignores features that are not in training", func(t *testing.T) {
		f := []bayes.Featurer{CookieF{"plain"}, ShapeF{"square"}}

		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxLabel, Jar1)
		assert.Equal(t, p.MaxOdds, 2.0)
	})

	t.Run("breaks on unknown features", func(t *testing.T) {
		f := []bayes.Featurer{UnknownF{}}
		_, err := nb.PosteriorOdds(f)
		assert.Equal(t, err.Error(), "All features are unknown")
	})

	t.Run("ignores uknown features", func(t *testing.T) {
		f := []bayes.Featurer{CookieF{"plain"}, UnknownF{}}
		p, err := nb.PosteriorOdds(f)
		assert.Nil(t, err)
		assert.Equal(t, p.MaxOdds, 2.0)
		assert.Equal(t, p.MaxLabel, Jar1)
	})
}

func TestPredict3Labels(t *testing.T) {
	lfs := threeCookieJarsFeatures()
	nb := bayes.TrainNB(lfs)

	t.Run("calculcates with 3 labels", func(t *testing.T) {
		p, err := nb.PosteriorOdds([]bayes.Featurer{CookieF{"chocolate"},
			CookieF{"chocolate"}})
		if err != nil {
			panic(err)
		}
		assert.Equal(t, p.MaxLabel, Jar3)
		assert.InDelta(t, p.MaxOdds, 4.479, 0.001)
	})

	t.Run("can calculate for 0 frequency", func(t *testing.T) {
		p, err := nb.PosteriorOdds([]bayes.Featurer{CookieF{"plain"}})
		assert.Nil(t, err)
		assert.Equal(t, p.MaxLabel, Jar1)
		assert.InDelta(t, p.MaxOdds, 2.0, 0.01)
	})
}

type Label int

const (
	Jar1 Label = iota
	Jar2
	Jar3
	Helicopter
	Surething
	Lots
	Little
)

var labels = []string{"Jar1", "Jar2", "Jar3", "Helicopter", "Surething", "Lots",
	"Little"}

var labelsDict = func() map[string]bayes.Labeler {
	res := make(map[string]bayes.Labeler)
	for i, v := range labels {
		res[v] = Label(i)
	}
	return res
}()

func (l Label) String() string {
	return labels[l]
}

type CookieF struct {
	kind string
}

func (c CookieF) Name() bayes.FeatureName {
	return featureName(c)
}

func (c CookieF) Value() bayes.FeatureValue {
	return bayes.FeatureValue(c.kind)
}

type ShapeF struct {
	kind string
}

func (s ShapeF) Name() bayes.FeatureName {
	return featureName(s)
}

func (s ShapeF) Value() bayes.FeatureValue {
	return bayes.FeatureValue(s.kind)
}

func featureName(f bayes.Featurer) bayes.FeatureName {
	t := strings.Split(reflect.TypeOf(f).Name(), ".")
	return bayes.FeatureName(t[len(t)-1])
}

type UnknownF struct{}

func (u UnknownF) Name() bayes.FeatureName {
	return featureName(u)
}

func (u UnknownF) Value() bayes.FeatureValue {
	return bayes.FeatureValue("true")
}

// cookieJarsFeatures implements features from
// https://en.wikipedia.org/wiki/Bayesian_inference
// We change number of cookies in the second jar to 30, so
// prior odds are not 1:1.
// there are 2 jars with cookies. Jar1 has 10 chocolate and 30 vanilla cookies,
// jar2 has 15 of each. If a cookie is randomly taken from a random jar, what
// is a probability it came from the jar1?
func cookieJarsFeatures() []bayes.LabeledFeatures {
	bayes.RegisterLabel(labelsDict)
	var f1 CookieF
	var f2 ShapeF
	var lfs []bayes.LabeledFeatures

	for i := 1; i <= 40; i++ {
		f1 = CookieF{"chocolate"}
		if i > 10 {
			f1 = CookieF{"plain"}
		}
		f2 = ShapeF{"star"}
		lf := bayes.LabeledFeatures{
			Label:    Jar1,
			Features: []bayes.Featurer{f1, f2},
		}
		lfs = append(lfs, lf)
	}

	for i := 1; i <= 30; i++ {
		f1 = CookieF{"chocolate"}
		if i > 15 {
			f1 = CookieF{"plain"}
		}
		f2 = ShapeF{"round"}
		lf := bayes.LabeledFeatures{
			Label:    Jar2,
			Features: []bayes.Featurer{f1, f2},
		}
		lfs = append(lfs, lf)
	}
	return lfs
}

func threeCookieJarsFeatures() []bayes.LabeledFeatures {
	var f CookieF
	lfs := cookieJarsFeatures()
	for i := 1; i <= 40; i++ {
		f = CookieF{"chocolate"}
		lf := bayes.LabeledFeatures{
			Label:    Jar3,
			Features: []bayes.Featurer{f},
		}
		lfs = append(lfs, lf)
	}
	return lfs
}
