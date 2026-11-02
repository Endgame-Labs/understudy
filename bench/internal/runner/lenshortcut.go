package runner

// Length-shortcut diagnostic of the ladder. A training pool whose lead labels correlate with response length teaches the
// understudent "longer means yes"; if the test's labels do not, held-out agreement collapses (pilot #1). Per rung it reports,
// from LENGTHS ONLY (no text leaves the data directory):
//
//   - the lead's yes-rate by response-length quartile on the pool prefix and on the test (the quartile cut points are the
//     TEST's, so the two are comparable);
//   - the understudent's yes-rate by quartile on the test;
//   - the Spearman correlation of the understudent's predicted p(yes) with length on the test, minus the lead's. A large
//     positive difference says the understudent leans on length more than the lead does.

import (
	"encoding/json"
	"math"
	"sort"
	"unicode/utf8"

	"github.com/dorkitude/understudy/bench/internal/ds"
)

// LengthShortcut is one rung's length diagnostic.
type LengthShortcut struct {
	Cutoffs             [3]int      `json:"quartile_cutoffs_chars"` // the test's quartile cut points
	PoolN               [4]int      `json:"pool_n"`
	PoolLeadYes         [4]*float64 `json:"pool_lead_yes"`
	TestN               [4]int      `json:"test_n"`
	TestLeadYes         [4]*float64 `json:"test_lead_yes"`
	TestUnderstudentYes [4]*float64 `json:"test_understudent_yes"`
	// SpearmanLead and SpearmanUnderstudent are the rank correlations of p(yes) with length on the test; Delta is the
	// understudent's minus the lead's.
	SpearmanLead         *float64 `json:"spearman_lead"`
	SpearmanUnderstudent *float64 `json:"spearman_understudent"`
	Delta                *float64 `json:"spearman_delta"`
	// PoolSpearmanLead is the lead's p(yes)-length correlation on the pool prefix (the cue the understudent can learn).
	PoolSpearmanLead *float64 `json:"pool_spearman_lead"`
}

// ResponseChars is the number of characters of a state's "response" field, and whether it has one.
func ResponseChars(state any) (int, bool) {
	raw, ok := state.(json.RawMessage)
	if !ok {
		var err error
		if raw, err = json.Marshal(state); err != nil {
			return 0, false
		}
	}
	var st struct {
		Response *string `json:"response"`
	}
	if json.Unmarshal(raw, &st) != nil || st.Response == nil {
		return 0, false
	}
	return utf8.RuneCountInString(*st.Response), true
}

// ranks returns the average ranks (1-based) of x, ties sharing their mean rank.
func ranks(x []float64) []float64 {
	idx := make([]int, len(x))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return x[idx[a]] < x[idx[b]] })
	r := make([]float64, len(x))
	for i := 0; i < len(idx); {
		j := i
		for j+1 < len(idx) && x[idx[j+1]] == x[idx[i]] {
			j++
		}
		mean := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			r[idx[k]] = mean
		}
		i = j + 1
	}
	return r
}

// Spearman is the rank correlation of x and y (Pearson on average ranks). ok is false when it is undefined (fewer than two
// points, or one side constant).
func Spearman(x, y []float64) (float64, bool) {
	if len(x) != len(y) || len(x) < 2 {
		return 0, false
	}
	a, b := ranks(x), ranks(y)
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma /= float64(len(a))
	mb /= float64(len(b))
	var sab, saa, sbb float64
	for i := range a {
		sab += (a[i] - ma) * (b[i] - mb)
		saa += (a[i] - ma) * (a[i] - ma)
		sbb += (b[i] - mb) * (b[i] - mb)
	}
	if saa == 0 || sbb == 0 {
		return 0, false
	}
	return sab / math.Sqrt(saa*sbb), true
}

// LenPoint is one item for the diagnostic: its response length, the lead's p(yes), and (test only) the understudent's
// predicted p(yes) and label (HasU false when it made no prediction).
type LenPoint struct {
	Len     int
	LeadYes float64 // lead p(yes)
	LeadLbl int     // lead's argmax class
	UYes    float64 // understudent p(yes)
	ULbl    int
	HasU    bool
}

func rate(k, n int) *float64 {
	if n == 0 {
		return nil
	}
	return fptr(float64(k) / float64(n))
}

// ComputeLengthShortcut builds the diagnostic from the pool prefix's and the test's points. yes is the index of the "true"
// class. The quartile cut points come from the test lengths.
func ComputeLengthShortcut(pool, test []LenPoint, yes int) *LengthShortcut {
	if len(test) == 0 {
		return nil
	}
	lens := make([]int, len(test))
	for i, p := range test {
		lens[i] = p.Len
	}
	ls := &LengthShortcut{Cutoffs: quartiles(lens)}
	var pk, tk, uk, un [4]int
	for _, p := range pool {
		q := quartileOf(p.Len, ls.Cutoffs)
		ls.PoolN[q]++
		if p.LeadLbl == yes {
			pk[q]++
		}
	}
	for _, p := range test {
		q := quartileOf(p.Len, ls.Cutoffs)
		ls.TestN[q]++
		if p.LeadLbl == yes {
			tk[q]++
		}
		if p.HasU {
			un[q]++
			if p.ULbl == yes {
				uk[q]++
			}
		}
	}
	for q := 0; q < 4; q++ {
		ls.PoolLeadYes[q] = rate(pk[q], ls.PoolN[q])
		ls.TestLeadYes[q] = rate(tk[q], ls.TestN[q])
		ls.TestUnderstudentYes[q] = rate(uk[q], un[q])
	}
	var l, ll, uu, ul []float64
	for _, p := range test {
		l = append(l, float64(p.Len))
		ll = append(ll, p.LeadYes)
		if p.HasU {
			uu = append(uu, p.UYes)
			ul = append(ul, float64(p.Len))
		}
	}
	if r, ok := Spearman(ll, l); ok {
		ls.SpearmanLead = fptr(r)
	}
	if r, ok := Spearman(uu, ul); ok {
		ls.SpearmanUnderstudent = fptr(r)
	}
	if ls.SpearmanLead != nil && ls.SpearmanUnderstudent != nil {
		ls.Delta = fptr(*ls.SpearmanUnderstudent - *ls.SpearmanLead)
	}
	var pl, pp []float64
	for _, p := range pool {
		pl = append(pl, float64(p.Len))
		pp = append(pp, p.LeadYes)
	}
	if r, ok := Spearman(pp, pl); ok {
		ls.PoolSpearmanLead = fptr(r)
	}
	return ls
}

// yesIndex is the class index of the "true" option (1 for the usual false/true).
func yesIndex(opts []string) int {
	for i, o := range opts {
		if o == "true" {
			return i
		}
	}
	return min(1, len(opts)-1)
}

// poolPoints converts the lead-labeled pool prefix into diagnostic points (nil when the states have no response field).
func poolPoints(pool []TrainLine, yes int) []LenPoint {
	var out []LenPoint
	for _, t := range pool {
		n, ok := ResponseChars(t.State)
		if !ok || len(t.LeadProbs) <= yes {
			return nil
		}
		out = append(out, LenPoint{Len: n, LeadYes: t.LeadProbs[yes], LeadLbl: ds.Argmax(t.LeadProbs)})
	}
	return out
}

// testPoints converts the held-out stream and a rung's decisions into diagnostic points.
func testPoints(stream []ds.StreamItem, dec []ds.Decision, yes int) []LenPoint {
	by := map[string]ds.Decision{}
	for _, d := range dec {
		by[d.ID] = d
	}
	var out []LenPoint
	for _, s := range stream {
		n, ok := ResponseChars(s.State)
		if !ok || len(s.LeadProbs) <= yes {
			return nil
		}
		p := LenPoint{Len: n, LeadYes: s.LeadProbs[yes], LeadLbl: ds.Argmax(s.LeadProbs)}
		if d, ok := by[s.ID]; ok && d.UnderstudentLabel != nil && len(d.UnderstudentProbs) > yes {
			p.HasU, p.ULbl, p.UYes = true, *d.UnderstudentLabel, d.UnderstudentProbs[yes]
		}
		out = append(out, p)
	}
	return out
}

// quartiles returns the three quartile cut points (the values at ranks n/4, n/2, 3n/4 of the sorted lengths). A length
// below the first cut is in Q1, from the first up to the second in Q2, and so on; at or above the third it is in Q4.
func quartiles(lens []int) [3]int {
	s := append([]int(nil), lens...)
	sort.Ints(s)
	var q [3]int
	if len(s) == 0 {
		return q
	}
	for k := 1; k <= 3; k++ {
		q[k-1] = s[k*len(s)/4]
	}
	return q
}

// quartileOf returns 0..3 for a length against quartile cut points.
func quartileOf(x int, q [3]int) int {
	switch {
	case x < q[0]:
		return 0
	case x < q[1]:
		return 1
	case x < q[2]:
		return 2
	}
	return 3
}
