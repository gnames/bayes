package bayes_test

import (
	"reflect"
	"strings"

	. "github.com/gnames/bayes"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Bayes", func() {
	Describe("NewNaiveBayes()", func() {
		It("Init classifier with reasonable defaults", func() {
			nb := NewNaiveBayes()
			Expect(len(nb.Labels)).To(Equal(0))
			Expect(nb).ToNot(Equal(NaiveBayes{}))
		})

	})

	Describe("TrainNB", func() {
		It("takes training data and returns NB classifier", func() {
			var lfs []LabeledFeatures
			lfs = cookieJarsFeatures()
			nb := TrainNB(lfs)
			Expect(len(nb.Labels)).To(Equal(2))
			plain := nb.FeatureFreq[FeatureName("CookieF")][FeatureValue("plain")]
			Expect(plain[Label("jar1")]).To(Equal(30.0))
			chocolate := nb.FeatureFreq[FeatureName("CookieF")][FeatureValue("chocolate")]
			Expect(chocolate[Label("jar2")]).To(Equal(20.0))
			Expect(nb.LabelFreq[Label("jar1")]).To(Equal(40.0))
		})
	})

	Describe("Dump/Restore", func() {
		It("Dumps the content of NaiveBayes object to json file", func() {
			lfs := cookieJarsFeatures()
			nb := TrainNB(lfs)
			json := nb.Dump()
			Expect(string(json)[0:15]).To(Equal("{\n  \"labels\": ["))
			nb2 := NewNaiveBayes()
			nb2.Restore(json)
			Expect(nb2.Total).To(Equal(80.0))
		})
	})

	Describe("Predict()", func() {
		Context("Two labels", func() {
			It("Calculates posterior Probabilities", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs)
				p, err := nb.Predict([]Feature{CookieF{"plain"}})
				if err != nil {
					panic(err)
				}
				Expect(p.MaxOdds).To(Equal(1.5))
				Expect(p.MaxLabel).To(Equal(Label("jar1")))
			})

			It("Calculates multiple posterior Probabilities", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs)
				p, err := nb.Predict([]Feature{CookieF{"plain"}, CookieF{"plain"}})
				if err != nil {
					panic(err)
				}
				Expect(p.MaxOdds).To(Equal(2.25))
				Expect(p.MaxLabel).To(Equal(Label("jar1")))
			})

			It("compensates infinite odds with crude smoothing", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs)
				f := []Feature{CookieF{"chocolate"}, ShapeF{"star"}}

				p, err := nb.Predict(f)
				if err != nil {
					panic(err)
				}
				Expect(p.MaxLabel).To(Equal(Label("jar1")))
				Expect(p.MaxOdds).To(BeNumerically("~", 20, 1))
			})

			It("ignores features that are not in training", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs)
				f := []Feature{CookieF{"plain"}, ShapeF{"square"}}

				p, err := nb.Predict(f)
				Expect(err).ToNot(HaveOccurred())
				Expect(p.MaxLabel).To(Equal(Label("jar1")))
				Expect(p.MaxOdds).To(Equal(1.5))
			})

			It("breaks on unknown features", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs)
				f := []Feature{UnknownF{}}
				_, err := nb.Predict(f)
				Expect(err.Error()).To(Equal("All features are unknown"))
			})

			It("ignores uknown features", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs)
				f := []Feature{CookieF{"plain"}, UnknownF{}}
				p, err := nb.Predict(f)
				Expect(err).ToNot(HaveOccurred())
				Expect(p.MaxOdds).To(Equal(1.5))
				Expect(p.MaxLabel).To(Equal(Label("jar1")))
			})
		})

		Context("Three labels", func() {
			It("calculcates with 3 labels", func() {
				lfs := threeCookieJarsFeatures()
				nb := TrainNB(lfs)
				p, err := nb.Predict([]Feature{CookieF{"chocolate"},
					CookieF{"chocolate"}})
				if err != nil {
					panic(err)
				}
				Expect(p.MaxLabel).To(Equal(Label("jar3")))
				Expect(p.MaxOdds).To(BeNumerically("~", 3.555, 0.001))
			})
		})

		It("can calculate for 0 frequency", func() {
			lfs := threeCookieJarsFeatures()
			nb := TrainNB(lfs)
			p, err := nb.Predict([]Feature{CookieF{"plain"}})
			if err != nil {
				panic(err)
			}
			Expect(p.MaxLabel).To(Equal(Label("jar1")))
			Expect(p.MaxOdds).To(BeNumerically("~", 1.499, 0.01))
		})
	})
})

type CookieF struct {
	kind string
}

func (c CookieF) Name() FeatureName {
	return featureName(c)
}

func (c CookieF) Value() FeatureValue {
	return FeatureValue(c.kind)
}

type ShapeF struct {
	kind string
}

func (s ShapeF) Name() FeatureName {
	return featureName(s)
}

func (s ShapeF) Value() FeatureValue {
	return FeatureValue(s.kind)
}

func featureName(f Feature) FeatureName {
	t := strings.Split(reflect.TypeOf(f).Name(), ".")
	return FeatureName(t[len(t)-1])
}

type UnknownF struct{}

func (u UnknownF) Name() FeatureName {
	return featureName(u)
}

func (u UnknownF) Value() FeatureValue {
	return FeatureValue("true")
}

// cookieJarsFeatures implements features from
// https://en.wikipedia.org/wiki/Bayesian_inference
// there are 2 jars with cookies. Jar1 has 10 chocolate and 20 vanilla cookies,
// jar2 has 20 of each. If a cookie is randomly taken from a random jar, what
// is a probability it came from the jar1?
func cookieJarsFeatures() []LabeledFeatures {
	var f1 CookieF
	var f2 ShapeF
	var lfs []LabeledFeatures

	for i := 1; i <= 40; i++ {
		f1 = CookieF{"chocolate"}
		if i > 10 {
			f1 = CookieF{"plain"}
		}
		f2 = ShapeF{"star"}
		lf := LabeledFeatures{
			Label:    Label("jar1"),
			Features: []Feature{f1, f2},
		}
		lfs = append(lfs, lf)
	}

	for i := 1; i <= 40; i++ {
		f1 = CookieF{"chocolate"}
		if i > 20 {
			f1 = CookieF{"plain"}
		}
		f2 = ShapeF{"round"}
		lf := LabeledFeatures{
			Label:    Label("jar2"),
			Features: []Feature{f1, f2},
		}
		lfs = append(lfs, lf)
	}
	return lfs
}

func threeCookieJarsFeatures() []LabeledFeatures {
	var f CookieF
	lfs := cookieJarsFeatures()
	for i := 1; i <= 40; i++ {
		f = CookieF{"chocolate"}
		lf := LabeledFeatures{
			Label:    Label("jar3"),
			Features: []Feature{f},
		}
		lfs = append(lfs, lf)
	}
	return lfs
}
