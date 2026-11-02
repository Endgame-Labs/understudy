package report

import (
	"testing"

	"github.com/endgame-labs/understudy/bench/internal/ds"
)

func dec(id, by string, pYes float64) ds.Decision {
	return ds.Decision{ID: id, AnsweredBy: by, Probs: []float64{1 - pYes, pYes}}
}

func TestIsPerResponse(t *testing.T) {
	if !IsPerResponse([]string{"a-0001#1", "a-0001#2"}) || IsPerResponse([]string{"a-0001"}) || IsPerResponse(nil) || IsPerResponse([]string{"a#x"}) {
		t.Fatal("IsPerResponse")
	}
}

func TestItemLevel(t *testing.T) {
	// item i1 (N=3): truth slot 2. Lead picks 2 (right). System: all local, picks 2 (right).
	// item i2: truth slot 1. Lead picks 1 (right); system forwarded slot 3 and the understudent's slot 2 beats it: wrong.
	// item i3: truth slot 3. Lead picks 1 (wrong); system all local, picks 3 (right).
	lead := map[string][]float64{
		"i1#1": {0.9, 0.1}, "i1#2": {0.2, 0.8}, "i1#3": {0.95, 0.05},
		"i2#1": {0.3, 0.7}, "i2#2": {0.8, 0.2}, "i2#3": {0.9, 0.1},
		"i3#1": {0.4, 0.6}, "i3#2": {0.9, 0.1}, "i3#3": {0.8, 0.2},
	}
	truth := map[string]int{"i1#2": 1, "i2#1": 1, "i3#3": 1, "i1#1": 0, "i1#3": 0, "i2#2": 0, "i2#3": 0, "i3#1": 0, "i3#2": 0}
	// stream order, sub-items contiguous but given out of slot order for i1 to check sorting
	ds_ := []ds.Decision{
		dec("i1#2", "understudent", 0.9), dec("i1#1", "understudent", 0.1), dec("i1#3", "understudent", 0.1),
		dec("i2#1", "lead", 0.7), dec("i2#2", "understudent", 0.75), dec("i2#3", "lead", 0.1),
		dec("i3#1", "understudent", 0.2), dec("i3#2", "understudent", 0.1), dec("i3#3", "understudent", 0.6),
	}
	m := ItemLevel(ds_, lead, truth)
	if m == nil {
		t.Fatal("nil")
	}
	if m.Items != 3 || m.Responses != 3 {
		t.Fatalf("items %d responses %d", m.Items, m.Responses)
	}
	if want := 2.0 / 3; m.AccLead < want-1e-9 || m.AccLead > want+1e-9 {
		t.Fatalf("lead %v", m.AccLead)
	}
	if want := 2.0 / 3; m.AccSystem < want-1e-9 || m.AccSystem > want+1e-9 {
		t.Fatalf("system %v", m.AccSystem)
	}
	if m.FullyLocalN != 2 {
		t.Fatalf("fully local %d", m.FullyLocalN)
	}
	if *m.AccLeadLocal != 0.5 || *m.AccUnderstudentLocal != 1 || *m.AgreeLocal != 0.5 {
		t.Fatalf("local %v %v %v", *m.AccLeadLocal, *m.AccUnderstudentLocal, *m.AgreeLocal)
	}
	if *m.AccLeadNotLoc != 1 || *m.AccSystemNotLoc != 0 {
		t.Fatalf("not local %v %v", *m.AccLeadNotLoc, *m.AccSystemNotLoc)
	}
}

func TestItemLevelRejects(t *testing.T) {
	truth := map[string]int{"a#1": 1, "a#2": 1}
	lead := map[string][]float64{"a#1": {0, 1}, "a#2": {0, 1}}
	if ItemLevel([]ds.Decision{dec("a#1", "lead", 1), dec("a#2", "lead", 1)}, lead, truth) != nil {
		t.Fatal("two correct responses must be rejected")
	}
	if ItemLevel([]ds.Decision{dec("a", "lead", 1)}, lead, truth) != nil {
		t.Fatal("non per-response ids must be rejected")
	}
}
