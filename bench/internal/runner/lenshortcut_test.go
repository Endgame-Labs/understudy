package runner

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/dorkitude/understudy/bench/internal/ds"
)

func TestSpearman(t *testing.T) {
	if r, ok := Spearman([]float64{1, 2, 3, 4}, []float64{10, 20, 30, 40}); !ok || math.Abs(r-1) > 1e-12 {
		t.Fatalf("monotone: %v %v", r, ok)
	}
	if r, ok := Spearman([]float64{1, 2, 3, 4}, []float64{9, 7, 5, 1}); !ok || math.Abs(r+1) > 1e-12 {
		t.Fatalf("anti: %v", r)
	}
	// ties share their mean rank: x = [1,1,2,3], y = [1,2,3,4]; rank(x) = [1.5,1.5,3,4]
	r, ok := Spearman([]float64{1, 1, 2, 3}, []float64{1, 2, 3, 4})
	want := 0.9486832980505138 // Pearson of [1.5,1.5,3,4] and [1,2,3,4]
	if !ok || math.Abs(r-want) > 1e-9 {
		t.Fatalf("ties: got %v want %v", r, want)
	}
	if _, ok := Spearman([]float64{1, 1, 1}, []float64{1, 2, 3}); ok {
		t.Fatal("a constant side has no correlation")
	}
	if _, ok := Spearman([]float64{1}, []float64{1}); ok {
		t.Fatal("one point has no correlation")
	}
}

func TestResponseChars(t *testing.T) {
	n, ok := ResponseChars(json.RawMessage(`{"instruction":"i","response":"héllo wörld"}`))
	if !ok || n != 11 {
		t.Fatalf("%d %v (characters, not bytes)", n, ok)
	}
	if _, ok := ResponseChars(json.RawMessage(`{"instruction":"i"}`)); ok {
		t.Fatal("no response field")
	}
	if n, ok := ResponseChars(json.RawMessage(`{"response":""}`)); !ok || n != 0 {
		t.Fatal("an empty response is a response of length 0")
	}
}

// a pool where the lead says yes only to long responses, a test where the lead ignores length, and an understudent that
// learned the pool's cue
func TestLengthShortcutDetectsALengthCue(t *testing.T) {
	var pool, test []LenPoint
	for i := 0; i < 400; i++ {
		l := 10 + i*10
		yes := 0.1
		if l > 2000 {
			yes = 0.9
		}
		lbl := 0
		if yes > 0.5 {
			lbl = 1
		}
		pool = append(pool, LenPoint{Len: l, LeadYes: yes, LeadLbl: lbl})
	}
	for i := 0; i < 400; i++ {
		l := 10 + i*10
		lead := 0.1 + 0.8*float64((i*7919)%2) // alternates with i: unrelated to length
		lbl := 0
		if lead > 0.5 {
			lbl = 1
		}
		u := float64(l) / 4100 // the understudent's p(yes) grows with length
		ulbl := 0
		if u > 0.5 {
			ulbl = 1
		}
		test = append(test, LenPoint{Len: l, LeadYes: lead, LeadLbl: lbl, UYes: u, ULbl: ulbl, HasU: true})
	}
	ls := ComputeLengthShortcut(pool, test, 1)
	if ls == nil || ls.Delta == nil || ls.SpearmanUnderstudent == nil || ls.SpearmanLead == nil || ls.PoolSpearmanLead == nil {
		t.Fatalf("%+v", ls)
	}
	if *ls.SpearmanUnderstudent < 0.99 || math.Abs(*ls.SpearmanLead) > 0.1 || *ls.Delta < 0.9 || *ls.PoolSpearmanLead < 0.7 {
		t.Errorf("understudent %.3f lead %.3f delta %.3f pool %.3f", *ls.SpearmanUnderstudent, *ls.SpearmanLead, *ls.Delta, *ls.PoolSpearmanLead)
	}
	// quartile cut points are the test's, the shares by quartile cover every item, and the pool's yes-rate rises with length
	sum := 0
	for _, n := range ls.TestN {
		sum += n
	}
	if sum != 400 || ls.TestN[0] != 100 || ls.Cutoffs[0] != test[100].Len {
		t.Errorf("test quartiles %v %v", ls.TestN, ls.Cutoffs)
	}
	if *ls.PoolLeadYes[0] != 0 || *ls.PoolLeadYes[3] != 1 {
		t.Errorf("pool lead yes by quartile %v %v", *ls.PoolLeadYes[0], *ls.PoolLeadYes[3])
	}
	if *ls.TestUnderstudentYes[0] != 0 || *ls.TestUnderstudentYes[3] != 1 {
		t.Errorf("understudent yes by quartile")
	}
	// a test with no understudent predictions has no understudent numbers, and the diagnostic is text-free
	for i := range test {
		test[i].HasU = false
	}
	ls2 := ComputeLengthShortcut(pool, test, 1)
	if ls2.Delta != nil || ls2.SpearmanUnderstudent != nil || ls2.TestUnderstudentYes[0] != nil {
		t.Error("no predictions, no understudent numbers")
	}
	raw, _ := json.Marshal(ls)
	if strings.Contains(string(raw), "response") || strings.Contains(string(raw), "instruction") {
		t.Errorf("not text-free: %s", raw)
	}
}

func TestPointsFromStreamAndDecisions(t *testing.T) {
	st := func(resp string) json.RawMessage {
		b, _ := json.Marshal(map[string]string{"instruction": "i", "response": resp})
		return b
	}
	stream := []ds.StreamItem{{ID: "a#1", State: st("12345"), LeadProbs: []float64{0.2, 0.8}}, {ID: "a#2", State: st("1"), LeadProbs: []float64{0.9, 0.1}}}
	one := 1
	dec := []ds.Decision{{ID: "a#1", UnderstudentLabel: &one, UnderstudentProbs: []float64{0.3, 0.7}}, {ID: "a#2"}}
	pts := testPoints(stream, dec, 1)
	if len(pts) != 2 || pts[0].Len != 5 || !pts[0].HasU || pts[0].UYes != 0.7 || pts[0].LeadLbl != 1 || pts[1].HasU || pts[1].LeadLbl != 0 {
		t.Fatalf("%+v", pts)
	}
	pp := poolPoints([]TrainLine{{ID: "p#1", State: st("abc"), LeadProbs: []float64{0.4, 0.6}}}, 1)
	if len(pp) != 1 || pp[0].Len != 3 || pp[0].LeadLbl != 1 {
		t.Fatalf("%+v", pp)
	}
	// states without a response field (not a per-response dataset) give no diagnostic
	if testPoints([]ds.StreamItem{{ID: "x", State: json.RawMessage(`{"prompt":"p"}`), LeadProbs: []float64{0.5, 0.5}}}, nil, 1) != nil {
		t.Error("no response field should give nil")
	}
	if yesIndex([]string{"false", "true"}) != 1 || yesIndex([]string{"true", "false"}) != 0 {
		t.Error("yesIndex")
	}
}
