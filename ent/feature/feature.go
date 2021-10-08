package feature

type LabeledFeatures struct {
	Label
	Features []Feature
}

type Feature struct {
	Name  Name
	Value Val
}

type Label string
type Name string
type Val string
