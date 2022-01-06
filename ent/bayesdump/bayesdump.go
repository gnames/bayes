package bayesdump

// BayesDump is a printing/serializing friendly presentation of data from
// private fields of Bayes implementation.
// It contains everything needed for training the classifier.
type BayesDump struct {
	// Classes is a slice of classes used for deciding where to place new
	// data.
	Classes []string `json:"classes"`

	// CasesTotal is the number of entities collected during training.
	CasesTotal int `json:"casesTotal"`

	// ClassCases is the number of entities partitioned to their corresponding
	// classes during training.
	ClassCases map[string]int `json:"classCases"`

	// FeatureCases is the entities from training paritioned by separate
	// features.
	FeatureCases map[string]map[string]map[string]int `json:"featureCases"`
}
