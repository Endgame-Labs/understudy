package report

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dorkitude/understudy/bench/internal/ds"
	"github.com/dorkitude/understudy/bench/internal/runner"
)

func TestWriteLadderRun(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "SPEC.md"), nil, 0o644)
	dir := filepath.Join(root, "runs", "lad")
	os.MkdirAll(filepath.Join(dir, "t_per"), 0o755)
	ds.WriteJSON(filepath.Join(dir, "run.json"), runner.RunMeta{Run: "lad", Lead: "jev", Datasets: []string{"t_per"}, Modes: []string{"ladder"},
		Ladder: []int{200, 1000}, Systems: []string{"understudent-deep"}})
	agree, lo, lift := 0.87, 0.85, 0.09
	ds.WriteJSON(filepath.Join(dir, "t_per", "ladder.json"), runner.LadderResult{Dataset: "t_per", Pool: "t_big_per", Lead: "jev", Ladder: []int{200, 1000},
		PoolItems: 10000, PoolReady: 10000, TestItems: 1800, Bar: 0.9, Rungs: []runner.Rung{
			{System: "understudent-deep", N: 200, TrainMajority: 0.75, CVAgreement: &agree, HeldMajority: 0.78, HeldAgreement: &agree, HeldWilsonLow: &lo, HeldLift: &lift},
			{System: "understudent-deep", N: 1000, Skipped: "the pool has only 400 lead-labeled items (of 10000)"}}})
	res, err := Write(root, "lad", "")
	if err != nil || res == nil {
		t.Fatal(err)
	}
	md, err := os.ReadFile(filepath.Join(dir, "LADDER.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"t_big_per", "87.0%", "85.0%", "only 400 lead-labeled", "clears 0.90"} {
		if !strings.Contains(string(md), want) {
			t.Errorf("LADDER.md lacks %q:\n%s", want, md)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "results.json")); err != nil {
		t.Fatal(err)
	}
}

func TestLadderMarkdownLengthShortcut(t *testing.T) {
	y := func(a, b, c, d float64) [4]*float64 { return [4]*float64{&a, &b, &c, &d} }
	rho, d := 0.31, 0.27
	r := &LadderResults{Run: "r", Lead: "jev", Ladder: []int{200}, Datasets: []runner.LadderResult{{Dataset: "t_per", Pool: "t_real_per", Rungs: []runner.Rung{
		{System: "understudent-deep", N: 200, LengthShortcut: &runner.LengthShortcut{Cutoffs: [3]int{305, 800, 2027}, PoolLeadYes: y(0.02, 0.28, 0.37, 0.33),
			TestLeadYes: y(0.24, 0.31, 0.26, 0.20), TestUnderstudentYes: y(0.15, 0.39, 0.58, 0.68), SpearmanLead: &rho, SpearmanUnderstudent: &rho, Delta: &d}}}}}}
	md := LadderMarkdown(r)
	for _, want := range []string{"Length shortcut", "305 / 800 / 2027", "0.02 / 0.28 / 0.37 / 0.33", "0.15 / 0.39 / 0.58 / 0.68", "+0.270", "Δ Spearman"} {
		if !strings.Contains(md, want) {
			t.Errorf("LADDER.md lacks %q:\n%s", want, md)
		}
	}
}
