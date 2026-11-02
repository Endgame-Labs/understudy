package report

import (
	"math"
	"strings"
	"testing"

	"github.com/endgame-labs/understudy/bench/internal/ds"
)

func TestPercentileIsNearestRank(t *testing.T) {
	if Percentile(nil, 50) != nil {
		t.Fatal("no values, no percentile")
	}
	xs := []float64{5, 1, 3, 2, 4, 10, 9, 8, 7, 6}
	if got := *Percentile(xs, 50); got != 5 {
		t.Fatalf("p50 = %v", got)
	}
	if got := *Percentile(xs, 99); got != 10 {
		t.Fatalf("p99 = %v", got)
	}
	if got := *Percentile([]float64{7}, 99); got != 7 {
		t.Fatalf("one value: %v", got)
	}
}

func TestLeadCostFollowsTheMeasuredRates(t *testing.T) {
	// 3,400 bytes = 1,000 tokens + the 600-token prompt = 1,600 tokens at $0.042 per million
	c, ok := CallCost("jev", 3400)
	if !ok || math.Abs(c-1600*0.042/1e6) > 1e-12 {
		t.Fatalf("jev: %v %v", c, ok)
	}
	if r, _ := CallCost("rune", 3400); r != c {
		t.Fatalf("rune and jev cost the same: %v %v", r, c)
	}
	if _, ok := CallCost("otherlead", 3400); ok {
		t.Fatal("otherlead has no measured rate")
	}
}

func TestUsefulnessCoverageAgreementAndDollars(t *testing.T) {
	us := func(x float64) *float64 { return &x }
	var dec []ds.Decision
	var lead []int
	for i := 0; i < 100; i++ {
		d := ds.Decision{ID: "x", AnsweredBy: "lead", Label: 1, UnderstudentUS: us(1000)}
		if i < 50 { // the understudent answers half the items, and agrees on 48 of them
			d.AnsweredBy, d.UnderstudentUS = "understudent", us(float64(10+i))
			d.Label = 1
		}
		dec = append(dec, d)
		l := 1
		if i == 3 || i == 7 {
			l = 0
		}
		lead = append(lead, l)
	}
	bytes := make([]int, 100)
	for i := range bytes {
		bytes[i] = 3400
	}
	u := ComputeUsefulness(dec, lead, []float64{100, 200, 300}, bytes, "jev")
	if u.Coverage != 0.5 || u.LocalN != 50 {
		t.Fatalf("coverage %v n %d", u.Coverage, u.LocalN)
	}
	if math.Abs(*u.Agreement-0.96) > 1e-9 {
		t.Fatalf("agreement %v", *u.Agreement)
	}
	if lo := *u.AgreementLow; lo < 0.86 || lo > 0.88 { // Wilson lower bound of 48/50
		t.Fatalf("wilson low %v", lo)
	}
	if *u.UnderstudentUSP50 != 34 || *u.UnderstudentUSP99 != 59 || *u.ForwardUSP50 != 1000 {
		t.Fatalf("latency %v %v %v", *u.UnderstudentUSP50, *u.UnderstudentUSP99, *u.ForwardUSP50)
	}
	if *u.LeadMSP50 != 200 || *u.LeadMSP99 != 300 {
		t.Fatalf("lead ms %v %v", *u.LeadMSP50, *u.LeadMSP99)
	}
	want := 0.5 * 1600 * 0.042 / 1e6 * 1000
	if math.Abs(*u.SavedPer1k-want) > 1e-12 {
		t.Fatalf("saved %v, want %v", *u.SavedPer1k, want)
	}
	want1m := 0.5 * 1600 * 0.042
	if math.Abs(*u.SavedPer1M-want1m) > 1e-9 {
		t.Fatalf("saved per 1M %v, want %v", *u.SavedPer1M, want1m)
	}
	if n := ComputeUsefulness(dec, lead, nil, bytes, "otherlead"); n.SavedPer1k != nil || n.SavedPer1M != nil {
		t.Fatal("no rate, no dollars")
	}
}

func TestUsefulnessSectionCitesTheRatesSource(t *testing.T) {
	u := ComputeUsefulness([]ds.Decision{{ID: "a", AnsweredBy: "understudent", Label: 1}}, []int{1}, nil, []int{100}, "rune")
	r := &Results{Datasets: []DatasetResult{{Name: "d", Systems: []SystemResult{{System: "understudent-memo", Usefulness: &u}}}}}
	var b strings.Builder
	usefulnessMarkdown(&b, r)
	out := b.String()
	if !strings.Contains(out, "## Usefulness") || !strings.Contains(out, RatesSource) || !strings.Contains(out, "understudent-memo") {
		t.Fatalf("section: %s", out)
	}
}

func TestLeadLatenciesCoverOnlyTheEvaluatedIDs(t *testing.T) {
	recs := map[string]ds.LeadRecord{
		"seed1": {ID: "seed1", LatencyMS: 9000},
		"seed2": {ID: "seed2", LatencyMS: 8000},
		"a":     {ID: "a", LatencyMS: 100},
		"b":     {ID: "b", LatencyMS: 200},
		"c":     {ID: "c"}, // no latency recorded
	}
	stream := []string{"seed1", "seed2", "a", "b", "c"}
	got := LeadLatencies(stream[2:], recs) // split mode: the first two are training seeds
	if len(got) != 2 || got[0] != 100 || got[1] != 200 {
		t.Fatalf("latencies %v, want [100 200] (training rows excluded)", got)
	}
	if p := Percentile(got, 99); p == nil || *p != 200 {
		t.Fatalf("p99 %v, want 200", p)
	}
}
