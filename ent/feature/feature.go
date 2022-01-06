package feature

type ClassFeatures struct {
	Class
	Features []Feature
}

type Feature struct {
	Name
	Value
}

type Class string
type Name string
type Value string
