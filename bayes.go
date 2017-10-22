/*
Package bayes implements Naive Bayes trainer and classifier.  Naive BayeNas
theorem calculates a probability of a hypothesis from a prior knowledge
about the hypothesis, as well as the evidence that supports or diminishes the
probability of the hypothesis.  Prior knowledge can dramatically influence the
posterior probability of a hypothesis.  For example assuming that an adult bird
that cannot fly is a penguin is very unlikely in the northern hemisphere, but
is very likely in Antarctica. Bayes' theorem is often depicted as

	P(H|E) = P(H) * P(E|H) / P(E)

where H is our hypothesis, E is a new evidence, P(H) is a prior probability of
H to be true, P(E|H) is a known probability for the evidence when H is true,
P(E) is a known probability of E in all known cases. P(H|E) is a posterior
probability of a hypothesis H adjusted accordingly to the new evidence E.

Finding a probability that a hypothesis is true can be considered a
classification event.  Given prior knowledge and a new evidence we are able to
classify an entity to a hypothesis that has the highest posterior probability.

Using odds instead of probabilities

It is possible to represent Bayes theorem using odds. Odds describe how likely
a hypothesis is in comparison to all other possible hypotheses.

	odds = P(H) / (1 - P(H))

	P(H) = odds / (1 + odds)

Using odds allows us to simplify Bayes calculations

	oddsPosterior = oddsPrior * likelihood

where likelihood is

	likelihood = P(E|H)/P(E|H')

P(E|H') in this case is a known probability of an evidence when H is not true.
In case if we have several evidences that are independent from each other,
posterior odds can be calculated as a product of prior odds and all likelihoods
of all given evidences.

	oddsPosterior = oddsPrior * likelihood1 * likelihood2 * likelihood3 ...

Each subsequent evidence modifies prior odds. If evidences are not independent
(for example inability to fly and a propensity to nesting on the ground for
birds) they skew the outcome. In reality given evidences are quite often not
completely independent. Because of that Naive Bayes got its name. People who
apply it "naively" state that their evidences are completely independent from
each other. In practice Naive Bayes approach often shows good results in spite
of this known fallacy.

Some likelihoods are infinite. For example for modern penguins likelihood of
not flying is infinite (there are no penguins able to fly). To deal with such
likelihoods we introduce Laplace smoothing. It is equivalent to adding 'fake'
occurrence for every feature" to each hypothesis.

Danger of float values underflow

Given a very large number of evidences (for example a probability words
occurrence in large texts) it is possible to get in a situation when we
multiply a large number of values that are smaller than 1 leading to a result
that is extremely close to 0. Such result might underflow float64 type. In such
cases it is better to use logarithms of prior odds and evidences likelihoods

	log(oddsPosterior) = log(oddsPrior) + log(likelihood1) + log(likelihood2) ...

Training and prior odds

It is quite possible that while likelihoods of evidences are quite
representative in the training data the prior odds from the training are not.
As in the previous example an evidence that a bird cannot fly supports a
'penguin' hypothesis much better in Antarctica because odds to meet a
penguin there are much higher than in the northern hemisphere. Therefore we
give an ability to supply prior probability value at a classification event.

Terminology

In natural language processing `evidences` are often called `features`. We
follow the same convention in this package.

Hypotheses are often called `classes` or `labels`. Based on the outcome we
classify an entity (assign a label to the entity in other words). In this
package we use the term `label` for hypotheses.
*/
package bayes

// Label is a string representation of a "hypothesis" or "class" the NaiveBayes
// is aware about.
type Label string

// FeatureName is a name of a Feature
type FeatureName string

// Feature is an interface of an "evidence" we use for training a NaiveBayes
// classifier, or for the classification of an unknown entity.
type Feature interface {
	Name() FeatureName
}

// FeatureFreq is a map for collecting frequences of training featureset.
// FeatureFreq is used for calculating Likelyhoods of a NaiveBayes classifier.
type FeatureFreq map[FeatureName]map[Label]float64

// FeatureTotal is used for calculating multinomal likelihoods. For example
// if we are interested in calculating likelihood of feature `f` its
// likelihood would be
//   L = P(f|H)/P(f/H')
// where `H` is main "hypothesis" or "label" and `H'` is a combination of all
// other hypotheses.
type FeatureTotal map[FeatureName]float64

// LabelFreq is a collection of counts for every Label in the training dataset.
// this information allows to calculate prior odds for a Label.
type LabelFreq map[Label]float64

// LabeledFeatures are data used for supervised training of NaiveBayes
// algorithm.
type LabeledFeatures struct {
	Features []Feature
	Label
}

// OptionNB is a type for options supplied to NaiveBayes classifier. It can
// support either flags or parameterized options.
type OptionNB func(*NaiveBayes) error

// NewNaiveBayes is a constructor for NaiveBayes object. It initializes
// several important defaults, and sets options that modify behavior of
// the NaiveBayes object.
// Currently constructor supports the following options:
//
// `WithLaplaceSmoothing` --- sets `LaplaceSmoothing` option to `true`
func NewNaiveBayes(opts ...OptionNB) *NaiveBayes {
	nb := &NaiveBayes{
		LabelFreq:    make(map[Label]float64),
		FeatureFreq:  make(map[FeatureName]map[Label]float64),
		featureTotal: make(map[FeatureName]float64),
	}
	for _, o := range opts {
		err := o(nb)
		if err != nil {
			panic(err)
		}
	}
	return nb
}

/*
NaiveBayes is a classifier for assigning an entity represented by its features
to a label.
*/
type NaiveBayes struct {
	LaplaceSmoothing  bool
	LidstoneSmoothing float64
	Labels            []Label
	FeatureFreq
	LabelFreq
	currentLabelFreq  LabelFreq
	currentLabelTotal float64
	featureTotal      map[FeatureName]float64
	total             float64
}
