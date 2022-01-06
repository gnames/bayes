package posterior

import ft "github.com/gnames/bayes/ent/feature"

// Odds are calculated posterior odds to classify an entity according to
// all the used features it contains.
//
// MaxOdds provides the best odds calculated for an entity and
// MaxClass provide the category with the best odds. This class is
// the desired classification result.
type Odds struct {
	// ClassOdds provide odds for each class.
	ClassOdds map[ft.Class]float64

	// MaxClass is the class with the best odds.
	MaxClass ft.Class

	// MaxOdds is the odds of the MaxClass
	MaxOdds float64

	ClassCases
	Likelihoods
}

// ClassCases is the number of cases per each class. They are used for the
// final calculation of the prior odds.
type ClassCases map[ft.Class]int

// Likelihoods are the odds for each feachure for every class.
// The multiplication product of all odds is the final odds for a class.
type Likelihoods map[ft.Class]map[ft.Feature]float64
