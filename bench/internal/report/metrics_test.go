package report

import (
	"math"
	"testing"

	"github.com/endgame-labs/understudy/bench/internal/ds"
)

func TestKappa(t *testing.T) {
	if k := Kappa([]int{0, 1, 0, 1}, []int{0, 1, 0, 1}, 2); k == nil || *k != 1 {
		t.Fatalf("perfect: %v", k)
	}
	// po = 0.5, pe = 0.5 -> 0
	if k := Kappa([]int{0, 0, 1, 1}, []int{0, 1, 0, 1}, 2); k == nil || math.Abs(*k) > 1e-12 {
		t.Fatalf("chance: %v", *k)
	}
	// classic example: 20 items, a/b table [[7,3],[2,8]] -> po .75, pe .5 -> .5
	var a, b []int
	add := func(x, y, n int) {
		for i := 0; i < n; i++ {
			a, b = append(a, x), append(b, y)
		}
	}
	add(0, 0, 7)
	add(0, 1, 3)
	add(1, 0, 2)
	add(1, 1, 8)
	if k := Kappa(a, b, 2); math.Abs(*k-0.5) > 1e-9 {
		t.Fatalf("kappa %v", *k)
	}
	if Kappa([]int{1, 1}, []int{1, 1}, 2) != nil {
		t.Fatal("kappa defined when pe = 1")
	}
}

func ip(i int) *int { return &i }

func TestCompute(t *testing.T) {
	sym := "symmetric"
	us := 100.0
	lead := []int{0, 1, 0, 0, 1, 0}
	truth := []int{0, 1, 0, 1, 1, 0}
	dec := []ds.Decision{
		{ID: "1", AnsweredBy: "lead", Label: 0},
		{ID: "2", AnsweredBy: "lead", Label: 1, UnderstudentLabel: ip(0)},
		{ID: "3", AnsweredBy: "lead", Label: 0, UnderstudentLabel: ip(0), Eligible: true, Mode: &sym}, // forwarded audit
		{ID: "4", AnsweredBy: "understudent", Label: 0, UnderstudentLabel: ip(0), Eligible: true, Mode: &sym, UnderstudentUS: &us},
		{ID: "5", AnsweredBy: "understudent", Label: 0, UnderstudentLabel: ip(0), Eligible: true, Mode: &sym, Audited: true, UnderstudentUS: &us},
		{ID: "6", AnsweredBy: "understudent", Label: 0, UnderstudentLabel: ip(0), Eligible: true, Mode: &sym},
	}
	m := Compute(dec, lead, truth, 2, 0)
	if !m.Graduated || m.GraduatedAtStart || *m.FirstUnderstudentItem != 4 {
		t.Fatalf("graduation: %+v", m)
	}
	if m.Coverage != 0.5 || m.Coverage2ndHalf != 1 {
		t.Fatalf("coverage %v %v", m.Coverage, m.Coverage2ndHalf)
	}
	// local: items 4,5,6: lead 0,1,0 understudent 0,0,0 -> agreement 2/3, minority (lead 1) recall 0/1
	if m.LocalN != 3 || math.Abs(*m.LocalAgreement-2.0/3) > 1e-12 || *m.LocalMinorityRecal != 0 || m.LocalMinorityN != 1 {
		t.Fatalf("local: %+v", m)
	}
	// audits: item 3 (forwarded eligible, agrees) and item 5 (local audited, disagrees)
	if m.AuditN != 2 || *m.AuditAgreement != 0.5 {
		t.Fatalf("audits %d %v", m.AuditN, *m.AuditAgreement)
	}
	// lead calls: 3 forwarded + 1 audit
	if m.LeadCalls != 4 || m.CallsSaved != 2 {
		t.Fatalf("calls %d saved %d", m.LeadCalls, m.CallsSaved)
	}
	// lead acc: items 4 wrong -> 5/6; system: labels 0,1,0,0,0,0 vs truth 0,1,0,1,1,0 -> 4/6
	if math.Abs(m.AccLead-5.0/6) > 1e-12 || math.Abs(m.AccSystem-4.0/6) > 1e-12 {
		t.Fatalf("acc %v %v", m.AccLead, m.AccSystem)
	}
	// understudent after first fit: items 2..6, understudent 0 everywhere, truth 1,0,1,1,0 -> 2/5
	if m.UnderstudentN != 5 || math.Abs(*m.AccUnderstudent-0.4) > 1e-12 {
		t.Fatalf("understudent %d %v", m.UnderstudentN, *m.AccUnderstudent)
	}
	if *m.UnderstudentUSMean != 100 || m.Verdict != "harmful" {
		t.Fatalf("us %v verdict %s", *m.UnderstudentUSMean, m.Verdict)
	}
}

func TestVerdict(t *testing.T) {
	cases := []struct {
		m    Metrics
		want string
	}{
		{Metrics{Graduated: false, AccLead: .9, AccSystem: .9}, "correctly abstains"},
		{Metrics{Graduated: true, Coverage2ndHalf: .6, AccLead: .9, AccSystem: .895}, "takes over"},
		{Metrics{Graduated: true, Coverage2ndHalf: .4, AccLead: .9, AccSystem: .9}, "partial"},
		{Metrics{Graduated: true, Coverage2ndHalf: .6, AccLead: .9, AccSystem: .885}, "partial"},
		{Metrics{Graduated: true, Coverage2ndHalf: .9, AccLead: .9, AccSystem: .87}, "harmful"},
	}
	for _, c := range cases {
		if got := Verdict(c.m); got != c.want {
			t.Errorf("%+v: %s, want %s", c.m, got, c.want)
		}
	}
}

func TestWinners(t *testing.T) {
	f := func(x float64) *float64 { return &x }
	d := DatasetResult{Systems: []SystemResult{
		{System: "a", Mode: "replay", Metrics: &Metrics{Graduated: true, Coverage2ndHalf: .8, AccLead: .9, AccSystem: .85}}, // too inaccurate
		{System: "b", Mode: "replay", Metrics: &Metrics{Graduated: true, Coverage2ndHalf: .6, AccLead: .9, AccSystem: .895, UnderstudentUSMean: f(500)}},
		{System: "c", Mode: "replay", Metrics: &Metrics{Graduated: true, Coverage2ndHalf: .6, AccLead: .9, AccSystem: .9, UnderstudentUSMean: f(100)}},
		{System: "d", Mode: "replay", Metrics: &Metrics{Graduated: false, AccLead: .9, AccSystem: .9}},
		{System: "e", Mode: "split", Metrics: &Metrics{Graduated: true, Coverage2ndHalf: .99, AccLead: .9, AccSystem: .9}},
	}}
	w := Winners(d, "replay")
	if len(w) != 2 || w[0].System != "c" || w[1].System != "b" {
		t.Fatalf("%v", w)
	}
}
