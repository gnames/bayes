package output_test

import (
	"encoding/json"
	"testing"

	"github.com/gnames/bayes/ent/output"
	"github.com/stretchr/testify/assert"
)

func TestMarshal(t *testing.T) {
	assert := assert.New(t)

	odds := output.OddsDetails{
		"one":   44.0,
		"two":   0,
		"three": 51.00332,
		"four":  12.112,
	}
	res, err := json.Marshal(odds)
	assert.Nil(err)

	assert.Contains(string(res), "[{\"feature\":")
}

func TestUnmarshal(t *testing.T) {
	assert := assert.New(t)
	jsn := `
[
  {"feature":"three","odds":51.00332},
  {"feature":"one","odds":44},
  {"feature":"four","odds":12.112},
  {"feature":"two","odds":0}
]`

	res := output.OddsDetails{}

	err := json.Unmarshal([]byte(jsn), &res)
	assert.Nil(err)
	assert.Equal(0.0, res["two"])
}
