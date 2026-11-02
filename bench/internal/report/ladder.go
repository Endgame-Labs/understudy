package report

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/endgame-labs/understudy/bench/internal/ds"
	"github.com/endgame-labs/understudy/bench/internal/runner"
)

// LadderResults is results.json of a fixed-test learning-curve run (`run --train-ladder`).
type LadderResults struct {
	Run       string                `json:"run"`
	Lead      string                `json:"lead"`
	Ladder    []int                 `json:"ladder"`
	Generated string                `json:"generated"`
	Datasets  []runner.LadderResult `json:"datasets"`
}

func pc(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("%.1f%%", 100**x)
}

// LadderMarkdown renders the learning curves.
func LadderMarkdown(r *LadderResults) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Learning curve: %s (lead %s)\n\n", r.Run, r.Lead)
	fmt.Fprintf(&b, "Rungs %v. Each rung trains an understudent on the first N lead-labeled items of the training pool and scores the SAME fixed held-out dataset with a frozen judge (nothing in the held-out stream is learned from).\n\n", r.Ladder)
	b.WriteString("- **CV**: the understudent's own grouped cross-validation on the N pool items (groups are prompts; the sub-items of one prompt never straddle folds), with the Wilson lower bound (95%, n = N, items treated as independent).\n")
	b.WriteString("- **held-out agree**: the understudent's own label vs the lead's, over every held-out item (not only those it would answer); **majority** is the lead's majority-class share on the held-out items; **lift** = agree − majority; **Wilson low** is the 95% lower bound of held-out agreement; **clears 0.90** says whether that bound is ≥ 0.90.\n")
	b.WriteString("- **coverage**: the share of held-out items the graduated understudent answered itself; **local agree** is its agreement with the lead on those.\n\n")
	for _, d := range r.Datasets {
		fmt.Fprintf(&b, "## %s\n\ntraining pool `%s` (%d items, %d lead-labeled prefix); held-out %d items.\n\n", d.Dataset, d.Pool, d.PoolItems, d.PoolReady, d.TestItems)
		b.WriteString("| system | N | train majority | CV agree | CV Wilson low | CV cov | mode | held-out agree | majority | lift | Wilson low | clears 0.90 | coverage | local agree | local Wilson low |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, x := range d.Rungs {
			if x.Skipped != "" || x.Error != "" {
				why := x.Skipped
				if x.Error != "" {
					why = "ERROR: " + x.Error
				}
				fmt.Fprintf(&b, "| %s | %d | %s |\n", x.System, x.N, why)
				continue
			}
			mode := "–"
			if x.Mode != nil {
				mode = *x.Mode
			}
			clears := "no"
			if x.ClearsBar {
				clears = "**yes**"
			}
			fmt.Fprintf(&b, "| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", x.System, x.N, pc(&x.TrainMajority),
				pc(x.CVAgreement), pc(x.CVWilsonLow), pc(x.CVCoverage), mode, pc(x.HeldAgreement), pc(&x.HeldMajority), pc(x.HeldLift),
				pc(x.HeldWilsonLow), clears, pc(&x.Coverage), pc(x.LocalAgreement), pc(x.LocalWilsonLow))
		}
		b.WriteString("\n")
		b.WriteString(lengthShortcutMarkdown(d))
	}
	return b.String()
}

func yesq(x [4]*float64) string {
	var p []string
	for _, v := range x {
		if v == nil {
			p = append(p, "–")
		} else {
			p = append(p, fmt.Sprintf("%.2f", *v))
		}
	}
	return strings.Join(p, " / ")
}

func sp(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("%+.3f", *x)
}

// lengthShortcutMarkdown renders the length-shortcut diagnostic of a dataset's rungs ("" when none has it).
func lengthShortcutMarkdown(d runner.LadderResult) string {
	var rows []runner.Rung
	for _, x := range d.Rungs {
		if x.LengthShortcut != nil {
			rows = append(rows, x)
		}
	}
	if len(rows) == 0 {
		return ""
	}
	var b strings.Builder
	cut := rows[0].LengthShortcut.Cutoffs
	fmt.Fprintf(&b, "### Length shortcut: %s\n\n", d.Dataset)
	fmt.Fprintf(&b, "Response-length quartiles use the TEST's cut points (%d / %d / %d characters); lengths only, no text. Yes-rates are listed Q1 (shortest) / Q2 / Q3 / Q4. **Δ Spearman** = Spearman(understudent p(yes), length) − Spearman(lead p(yes), length) on the test: positive and large means the understudent leans on length more than the lead does. A pool whose lead yes-rate rises with length (pool lead yes) teaches that cue.\n\n", cut[0], cut[1], cut[2])
	b.WriteString("| system | N | pool lead yes | pool lead ρ | test lead yes | test understudent yes | test lead ρ | test understudent ρ | **Δ Spearman** |\n|---|---|---|---|---|---|---|---|---|\n")
	for _, x := range rows {
		l := x.LengthShortcut
		fmt.Fprintf(&b, "| %s | %d | %s | %s | %s | %s | %s | %s | **%s** |\n", x.System, x.N, yesq(l.PoolLeadYes), sp(l.PoolSpearmanLead), yesq(l.TestLeadYes), yesq(l.TestUnderstudentYes), sp(l.SpearmanLead), sp(l.SpearmanUnderstudent), sp(l.Delta))
	}
	b.WriteString("\n")
	return b.String()
}

// BuildLadder collects every dataset's ladder.json of a ladder run.
func BuildLadder(root, run string, meta runner.RunMeta) (*LadderResults, error) {
	runDir := filepath.Join(root, "runs", run)
	r := &LadderResults{Run: run, Lead: meta.Lead, Ladder: meta.Ladder, Generated: time.Now().UTC().Format(time.RFC3339)}
	for _, name := range meta.Datasets {
		l, err := runner.LadderJSON(runDir, name)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		r.Datasets = append(r.Datasets, *l)
	}
	return r, nil
}

// writeLadder writes results.json and LADDER.md of a ladder run and, for a private data dir, verifies they are text-free.
func writeLadder(root, run, dataDir string, meta runner.RunMeta) (*LadderResults, error) {
	r, err := BuildLadder(root, run, meta)
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(root, "runs", run)
	if err := ds.WriteJSON(filepath.Join(dir, "results.json"), r); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(dir, "LADDER.md"), []byte(LadderMarkdown(r)), 0o644); err != nil {
		return nil, err
	}
	if meta.NoText || runner.IsPrivateDataDir(meta.DataDir) {
		if err := runner.StripChecklistsFile(filepath.Join(dir, "results.json")); err != nil {
			return r, err
		}
		for _, d := range r.Datasets {
			for _, dsName := range []string{d.Dataset, d.Pool} {
				items := filepath.Join(runner.DataPath(root, meta.DataDir, dsName), "items.jsonl")
				if err := runner.VerifyNoText(filepath.Join(dir, d.Dataset), items); err != nil {
					return r, err
				}
				if err := runner.ScanFiles([]string{filepath.Join(dir, "results.json"), filepath.Join(dir, "LADDER.md"), filepath.Join(dir, "run.json")}, items); err != nil {
					return r, err
				}
			}
		}
	}
	return r, nil
}
