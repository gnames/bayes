package output

// Output is a printing/serializing friendly presentation of data from
// private fields of Bayes implementation.
type Output struct {
	// Labels is a slice of labels (classes) used for deciding where to place new
	// data.
	Labels []string `json:"labels"`
	// CasesTotal is the number of entities collected during training.
	CasesTotal float64 `json:"casesTotal"`
	// LabelCases is the number of entities partitioned to their corresponding
	// labels during training.
	LabelCases map[string]float64 `json:"labelCases"`
	// FeatureCases is the entities from training paritioned by separate
	// features.
	FeatureCases map[string]map[string]map[string]float64 `json:"featureCases"`
}
