# bayes [![Build Status][travis-img]][travis] [![Doc Status][doc-img]][doc]

A simple implementation of Naive Bayes classifier. More details are in [docs].

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
