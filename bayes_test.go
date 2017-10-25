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
			Expect(nb.LaplaceSmoothing).To(Equal(false))
		})

	})

	Describe("TrainNB", func() {
		It("takes training data and returns NB classifier", func() {
			var lfs []LabeledFeatures
			lfs = cookieJarsFeatures()
			nb := TrainNB(lfs)
			Expect(len(nb.Labels)).To(Equal(2))
			Expect(nb.FeatureFreq[FeatureName("PlainF")][Label("jar1")]).To(Equal(30.0))
			Expect(nb.FeatureFreq[FeatureName("ChocolateF")][Label("jar2")]).
				To(Equal(20.0))
			Expect(nb.LabelFreq[Label("jar1")]).To(Equal(40.0))
		})
		It("can return Laplace smoothing", func() {
			var lfs []LabeledFeatures
			lfs = cookieJarsFeatures()
			nb := TrainNB(lfs, WithLaplaceSmoothing)
			Expect(len(nb.Labels)).To(Equal(2))
			Expect(nb.FeatureFreq[FeatureName("PlainF")][Label("jar1")]).To(Equal(31.0))
			Expect(nb.FeatureFreq[FeatureName("ChocolateF")][Label("jar2")]).
				To(Equal(21.0))
			Expect(nb.LabelFreq[Label("jar1")]).To(Equal(42.0))
		})
	})

	Describe("Dump/Restore", func() {
		It("Dumps the content of NaiveBayes object to json file", func() {
			lfs := cookieJarsFeatures()
			nb := TrainNB(lfs)
			json := nb.Dump()
			Expect(string(json)[0:20]).To(Equal("{\n  \"laplace\": false"))
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
				p, err := nb.Predict([]Feature{PlainF{}})
				if err != nil {
					panic(err)
				}
				Expect(p.MaxOdds).To(Equal(1.5))
				Expect(p.MaxLabel).To(Equal(Label("jar1")))
			})

			It("Calculates multiple features", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs)
				f := []Feature{ChocolateF{}, ChocolateF{}, ChocolateF{}, ChocolateF{}}
				p, err := nb.Predict(f)
				if err != nil {
					panic(err)
				}
				Expect(p.MaxLabel).To(Equal(Label("jar2")))
				Expect(p.MaxOdds).To(Equal(16.0))
			})

			It("Calculcates using smoothed training set", func() {
				lfs := cookieJarsFeatures()
				nb := TrainNB(lfs, WithLaplaceSmoothing)
				p, err := nb.Predict([]Feature{PlainF{}})
				if err != nil {
					panic(err)
				}
				Expect(p.MaxLabel).To(Equal(Label("jar1")))
				Expect(p.MaxOdds).To(BeNumerically("~", 1.476, 0.001))
			})
		})

		Context("Three labels", func() {
			It("Calculcates with 3 labels", func() {
				lfs := threeCookieJarsFeatures()
				nb := TrainNB(lfs)
				p, err := nb.Predict([]Feature{ChocolateF{}, ChocolateF{}})
				if err != nil {
					panic(err)
				}
				Expect(p.MaxLabel).To(Equal(Label("jar3")))
				Expect(p.MaxOdds).To(BeNumerically("~", 3.555, 0.001))
			})
		})

		It("Can calculate for 0 frequency", func() {
			lfs := threeCookieJarsFeatures()
			nb := TrainNB(lfs)
			p, err := nb.Predict([]Feature{PlainF{}})
			if err != nil {
				panic(err)
			}
			Expect(p.MaxLabel).To(Equal(Label("jar1")))
			Expect(p.MaxOdds).To(BeNumerically("~", 1.499, 0.01))
		})

		It("Can calculate with Lidstone smoothing", func() {
			lfs := threeCookieJarsFeatures()
			nb := TrainNB(lfs, WithLidstoneSmoothing(0.5))
			p, err := nb.Predict([]Feature{PlainF{}})
			if err != nil {
				panic(err)
			}
			Expect(p.MaxLabel).To(Equal(Label("jar1")))
			Expect(p.MaxOdds).To(BeNumerically("~", 1.452, 0.01))
		})
	})
})

type ChocolateF struct{}
type PlainF struct{}

func (c ChocolateF) Name() FeatureName {
	return featureName(c)
}

func (p PlainF) Name() FeatureName {
	return featureName(p)
}

func featureName(f Feature) FeatureName {
	t := strings.Split(reflect.TypeOf(f).Name(), ".")
	return FeatureName(t[len(t)-1])
}

// cookieJarsFeatures implements features from
// https://en.wikipedia.org/wiki/Bayesian_inference
// there are 2 jars with cookies. Jar1 has 10 chocolate and 20 vanilla cookies,
// jar2 has 20 of each. If a cookie is randomly taken from a random jar, what
// is a probability it came from the jar1?
func cookieJarsFeatures() []LabeledFeatures {
	var f Feature
	lfs := []LabeledFeatures{
		{Label: Label("jar1")},
		{Label: Label("jar2")},
	}

	for i := 1; i <= 40; i++ {
		f = ChocolateF{}
		if i > 10 {
			f = PlainF{}
		}
		lfs[0].Features = append(lfs[0].Features, f)
	}

	for i := 1; i <= 40; i++ {
		f = ChocolateF{}
		if i > 20 {
			f = PlainF{}
		}
		lfs[1].Features = append(lfs[1].Features, f)
	}
	return lfs
}

func threeCookieJarsFeatures() []LabeledFeatures {
	var f Feature
	lfs := cookieJarsFeatures()
	lfs = append(lfs, LabeledFeatures{Label: Label("jar3")})
	for i := 1; i <= 40; i++ {
		f = ChocolateF{}
		lfs[2].Features = append(lfs[2].Features, f)
	}
	return lfs
}
