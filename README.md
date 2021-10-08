# bayes [![Build Status][travis-img]][travis] [![Doc Status][doc-img]][doc]

An implementation of Naive Bayes classifier. More details are in [docs].

## Usage

This package allows to classify a new entity into one or another category (label)
according to features of the entity. The algorithm uses known data to calculate
a weight of each feature for each category.

```go
func Example() {
	// there are two jars of cookies, they are our training set.
	// Cookies have be round or star-shaped.
	// There are plain or chocolate chips cookies.
	jar1 := ft.Label("Jar1")
	jar2 := ft.Label("Jar2")

	// Every labeled feature-set provides data for one cookie. It tells
	// what jar has the cookie, what its kind and shape.
	cookie1 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("round")},
		},
	}
	cookie2 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie3 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie4 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("round")},
		},
	}
	cookie5 := ft.LabeledFeatures{
		Label: jar1,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("plain")},
			{Name: ft.Name("shape"), Value: ft.Val("round")},
		},
	}
	cookie6 := ft.LabeledFeatures{
		Label: jar2,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie7 := ft.LabeledFeatures{
		Label: jar2,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}
	cookie8 := ft.LabeledFeatures{
		Label: jar2,
		Features: []ft.Feature{
			{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
			{Name: ft.Name("shape"), Value: ft.Val("star")},
		},
	}

	lfs := []ft.LabeledFeatures{
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
		{Name: ft.Name("kind"), Value: ft.Val("chocolate")},
		{Name: ft.Name("shape"), Value: ft.Val("star")},
	}

	res, err := nb.PosteriorOdds(aCookie)
	if err != nil {
		fmt.Println(err)
	}

	// it is more likely to that a random cookie comes from Jar1, but
	// for chocolate and star-shaped cookie it is more likely to come from
	// Jar2.
	fmt.Printf("Prior odds for Jar1 are %0.2f\n", oddsPrior)
	fmt.Printf("The cookie came from %s, with odds %0.2f\n", res.MaxLabel, res.MaxOdds)
	// Output:
	// Prior odds for Jar1 are 1.67
	// The cookie came from Jar2, with odds 7.50
}
```

## Development

### Testing

Install [ginkgo], a [BDD] testing framefork for Go.

```bash
go get github.com/onsi/ginkgo/ginkgo
go get github.com/onsi/gomega
```

To run tests go to root directory of the project and run

```bash
ginkgo

#or

go test
```

## Other implementations:

[Go][go-bayes],
[Java][java-bayes],
[Python][py-bayes],
[R][r-bayes],
[Ruby][ruby-bayes]

[travis-img]: https://travis-ci.org/gnames/bayes.svg?branch=master
[travis]: https://travis-ci.org/gnames/bayes
[doc-img]: https://godoc.org/github.com/gnames/bayes?status.png
[doc]: https://godoc.org/github.com/gnames/bayes
[BDD]: https://en.wikipedia.org/wiki/Behavior-driven_development
[ginkgo]: https://github.com/onsi/ginkgo#set-me-up
[docs]: https://godoc.org/github.com/gnames/bayes
[r-bayes]: https://CRAN.R-project.org/package=naivebayes
[py-bayes]: http://www.nltk.org/api/nltk.classify.html#nltk.classify.naivebayes.NaiveBayesClassifier
[java-bayes]: https://github.com/haifengl/smile/blob/master/core/src/main/java/smile/classification/NaiveBayes.java
[go-bayes]: https://github.com/cdipaolo/goml/blob/master/text/bayes.go
[ruby-bayes]: https://github.com/oasic/nbayes
