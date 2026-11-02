package runner

// Fixed-test learning curve ("ladder", `understudy-bench run --train-ladder 200,1000,...`): for each rung N, seed an
// understudent with the first N lead-labeled items of a training POOL dataset (e.g. judge_rb2_safety_big_per), then score
// a different, fixed TEST dataset (the original judge_rb2_safety_per) with the judge frozen (`understudy judge
// --frozen`: nothing in the test stream is learned from), so every rung faces exactly the same held-out items.
//
// Per rung it reports the understudent's grouped cross-validation on the pool prefix (groups are prompts: the sub-items
// `<item>#<k>` of one prompt never straddle folds), the held-out agreement with the lead, the held-out majority share,
// the coverage (share of held-out items the understudent answered itself) and Wilson lower bounds against the 0.90 bar.

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/dorkitude/understudy/bench/internal/ds"
)

// LadderBar is the agreement bar the Wilson lower bounds are compared with.
const LadderBar = 0.90

// ParseLadder parses "200,1000,2000" into strictly ascending positive rungs.
func ParseLadder(s []string) ([]int, error) {
	var out []int
	for _, f := range s {
		for _, part := range strings.Split(f, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			var n int
			if _, err := fmt.Sscanf(part, "%d", &n); err != nil || n <= 0 || fmt.Sprint(n) != part {
				return nil, fmt.Errorf("bad ladder rung %q (want positive integers)", part)
			}
			if len(out) > 0 && n <= out[len(out)-1] {
				return nil, fmt.Errorf("ladder rungs must be strictly ascending (%d after %d)", n, out[len(out)-1])
			}
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("empty ladder")
	}
	return out, nil
}

// DefaultPool names the training pool of a test dataset: "x_per" -> "x_big_per", "x" -> "x_big".
func DefaultPool(test string) string {
	if strings.HasSuffix(test, "_per") {
		return strings.TrimSuffix(test, "_per") + "_big_per"
	}
	return test + "_big"
}

// Wilson returns the 95% Wilson interval (z = 1.96, as the understudent's graduation) for k successes in m trials.
func Wilson(k, m int) (lo, hi float64) {
	if m == 0 {
		return 0, 1
	}
	z := 1.96
	kk, mm := float64(k), float64(m)
	p := kk / mm
	c := (p + z*z/(2*mm)) / (1 + z*z/mm)
	h := z * math.Sqrt(p*(1-p)/mm+z*z/(4*mm*mm)) / (1 + z*z/mm)
	return c - h, c + h
}

// Rung is one (system, N) point of a learning curve.
type Rung struct {
	System  string `json:"system"`
	N       int    `json:"n"`
	Skipped string `json:"skipped,omitempty"`
	Error   string `json:"error,omitempty"`

	TrainItems    int      `json:"train_items"`
	DedupeDrops   int      `json:"train_dedupe_drops"` // pool items among the first N whose state repeats an earlier one (the fit keeps one)
	FitN          int      `json:"fit_n"`              // examples the fit actually used (must be train_items - train_dedupe_drops)
	TrainGroups   int      `json:"train_groups"`
	TrainMajority float64  `json:"train_majority_share"`
	CVAgreement   *float64 `json:"cv_agreement"`
	CVWilsonLow   *float64 `json:"cv_wilson_low"`
	CVCoverage    *float64 `json:"cv_coverage"`
	CVMinority    *float64 `json:"cv_minority_recall"`
	Mode          *string  `json:"mode"`
	Readout       *string  `json:"readout"`
	WhyNot        any      `json:"why_not,omitempty"`

	HeldN             int      `json:"heldout_n"`
	HeldMajority      float64  `json:"heldout_majority_share"`
	HeldAgreement     *float64 `json:"heldout_agreement"` // the understudent's own label vs the lead's, on every held-out item
	HeldWilsonLow     *float64 `json:"heldout_wilson_low"`
	HeldLift          *float64 `json:"heldout_lift_over_majority"`
	HeldAccTruth      *float64 `json:"heldout_acc_vs_truth"`
	ClearsBar         bool     `json:"heldout_wilson_low_clears_bar"`
	Coverage          float64  `json:"coverage"` // share of held-out items the understudent answered itself
	LocalN            int      `json:"local_n"`
	LocalAgreement    *float64 `json:"local_agreement"`
	LocalWilsonLow    *float64 `json:"local_wilson_low"`
	LocalClearsBar    bool     `json:"local_wilson_low_clears_bar"`
	TrainWallS        float64  `json:"train_wall_s"`
	JudgeWallS        float64  `json:"judge_wall_s"`
	Bar               float64  `json:"bar"`
	HeldGroupsApprox  int      `json:"heldout_groups"`
	HeldPrefixWarning string   `json:"note,omitempty"`
	// LengthShortcut is the response-length diagnostic (lengths only); nil when the states have no response field.
	LengthShortcut *LengthShortcut `json:"length_shortcut,omitempty"`
}

// LadderResult is runs/<run>/<ds>/ladder.json.
type LadderResult struct {
	Dataset   string `json:"dataset"`
	Pool      string `json:"pool"`
	Lead      string `json:"lead"`
	Ladder    []int  `json:"ladder"`
	PoolReady int    `json:"pool_labeled_prefix"` // items usable as the ladder's pool: the lead-labeled prefix minus the excluded prompts
	// PoolRawReady is the lead-labeled prefix before excluding prompts near-duplicate to a test prompt; PoolExcluded
	// names the excluded parent items (ids only, no text) and PoolExcludedItems counts their sub-items.
	PoolRawReady      int      `json:"pool_labeled_prefix_raw"`
	PoolExcluded      []string `json:"pool_excluded_parents,omitempty"`
	PoolExcludedItems int      `json:"pool_excluded_items"`
	PoolItems         int      `json:"pool_items"`
	TestItems         int      `json:"test_items"`
	Bar               float64  `json:"bar"`
	Time              string   `json:"time"`
	Rungs             []Rung   `json:"rungs"`
}

func fptr(x float64) *float64 { return &x }

// trainOut is `understudy train`'s stdout.
type trainOut struct {
	N           int      `json:"n"`
	CVAgreement *float64 `json:"cv_agreement"`
	CVMinority  *float64 `json:"cv_minority_recall"`
	CVCoverage  *float64 `json:"cv_coverage"`
	Mode        *string  `json:"mode"`
	Readout     *string  `json:"readout"`
	WhyNot      any      `json:"why_not"`
}

// ScoreHeldOut fills a rung's held-out numbers from the frozen judge's decisions. lead and truth map item id -> class.
func ScoreHeldOut(r *Rung, dec []ds.Decision, lead, truth map[string]int, k int) {
	r.Bar = LadderBar
	r.HeldN = len(dec)
	if len(dec) == 0 {
		return
	}
	counts := make([]int, k)
	groups := map[string]bool{}
	var agree, predicted, local, localAgree, truthRight int
	for _, d := range dec {
		l := lead[d.ID]
		counts[l]++
		parent, _ := splitParent(d.ID)
		groups[parent] = true
		if d.UnderstudentLabel != nil {
			predicted++
			if *d.UnderstudentLabel == l {
				agree++
			}
			if *d.UnderstudentLabel == truth[d.ID] {
				truthRight++
			}
		}
		if d.AnsweredBy == "understudent" {
			local++
			if d.Label == l {
				localAgree++
			}
		}
	}
	best := 0
	for _, c := range counts {
		best = max(best, c)
	}
	r.HeldMajority = float64(best) / float64(len(dec))
	r.HeldGroupsApprox = len(groups)
	r.Coverage = float64(local) / float64(len(dec))
	r.LocalN = local
	if local > 0 {
		r.LocalAgreement = fptr(float64(localAgree) / float64(local))
		lo, _ := Wilson(localAgree, local)
		r.LocalWilsonLow = fptr(lo)
		r.LocalClearsBar = lo >= LadderBar
	}
	if predicted > 0 {
		a := float64(agree) / float64(predicted)
		r.HeldAgreement = fptr(a)
		lo, _ := Wilson(agree, predicted)
		r.HeldWilsonLow = fptr(lo)
		r.ClearsBar = lo >= LadderBar
		r.HeldLift = fptr(a - r.HeldMajority)
		r.HeldAccTruth = fptr(float64(truthRight) / float64(predicted))
	}
}

func splitParent(id string) (string, bool) {
	if i := strings.LastIndexByte(id, '#'); i >= 0 {
		return id[:i], true
	}
	return id, false
}

// labeledPrefix returns the pool items in stream order that have a lead answer for their current state, stopping at
// the first one that does not (a pool whose lead labeling is still running is used up to there).
func labeledPrefix(poolDir, lead string) ([]TrainLine, int, error) {
	items, err := ds.ReadJSONL[ds.Item](filepath.Join(poolDir, "items.jsonl"))
	if err != nil {
		return nil, 0, err
	}
	recs, err := ds.ReadJSONL[ds.LeadRecord](ds.LeadPath(poolDir, lead))
	if err != nil {
		return nil, 0, err
	}
	by := map[string]ds.LeadRecord{}
	for _, r := range recs {
		by[r.ID] = r
	}
	var out []TrainLine
	for _, it := range items {
		r, ok := by[it.ID]
		if !ok || (r.StateSHA != "" && r.StateSHA != ds.StateSHA(it.State)) {
			break
		}
		out = append(out, TrainLine{ID: it.ID, State: it.State, LeadProbs: r.Probs})
	}
	return out, len(items), nil
}

// statePrompt returns the instruction of a state ("instruction", else "prompt").
func statePrompt(state any) string {
	raw, ok := state.(json.RawMessage)
	if !ok {
		raw, _ = json.Marshal(state)
	}
	var st struct {
		Instruction string `json:"instruction"`
		Prompt      string `json:"prompt"`
	}
	_ = json.Unmarshal(raw, &st)
	if st.Instruction != "" {
		return st.Instruction
	}
	return st.Prompt
}

// separatePool enforces that the training pool and the test set are disjoint. Any pool item sharing an id with a test
// item, or whose prompt equals a test prompt after normalization, rejects the run (an error). A pool prompt that is a
// near-duplicate of a test prompt (dedupe: 4-gram shingles, word-set Jaccard, edit similarity) is excluded together
// with every sub-item of its parent, WITHOUT renumbering: the kept items keep their ids and order, so the first N of the
// result is still the first N of the pool minus the excluded ones, nested across rungs. It returns the kept items and
// the excluded parent ids.
func separatePool(pool []TrainLine, test []ds.Item) (kept []TrainLine, excluded []string, err error) {
	testIDs := map[string]bool{}
	dd := newDedupe()
	testNorm := map[string]bool{}
	for _, it := range test {
		testIDs[it.ID] = true
		if p := statePrompt(it.State); p != "" {
			dd.Ban(p)
			testNorm[normPrompt(p)] = true
		}
	}
	verdict := map[string]string{} // prompt -> "" (ok) or the near-dup reason
	drop := map[string]bool{}      // parent ids
	for _, t := range pool {
		if testIDs[t.ID] {
			return nil, nil, fmt.Errorf("pool item %s has the same id as a test item: the pool and the test set must be disjoint", t.ID)
		}
		p := statePrompt(t.State)
		if p == "" {
			continue
		}
		if testNorm[normPrompt(p)] {
			return nil, nil, fmt.Errorf("pool item %s has the same prompt as a test item (after normalization): the pool and the test set must be disjoint", t.ID)
		}
		why, seen := verdict[p]
		if !seen {
			_, why = dd.Match(p)
			verdict[p] = why
		}
		if why != "" {
			parent, _ := splitParent(t.ID)
			drop[parent] = true
		}
	}
	for _, t := range pool {
		if parent, _ := splitParent(t.ID); drop[parent] {
			continue
		}
		kept = append(kept, t)
	}
	for _, t := range pool {
		if parent, _ := splitParent(t.ID); drop[parent] {
			if len(excluded) == 0 || excluded[len(excluded)-1] != parent {
				excluded = append(excluded, parent)
			}
		}
	}
	return kept, excluded, nil
}

// LadderName is the file suffix of a (system, rung): "understudent-deep-n1000".
func LadderName(sys string, n int) string { return fmt.Sprintf("%s-n%d", sys, n) }

func (c Config) poolFor(test string) string {
	if p, ok := c.Pools[test]; ok && p != "" {
		return p
	}
	return DefaultPool(test)
}

// ladderDataset runs every rung of every understudent system for one test dataset.
func (c Config) ladderDataset(ctx context.Context, name string) (err error) {
	testDir := DataPath(c.Root, c.DataDir, name)
	poolName := c.poolFor(name)
	poolDir := DataPath(c.Root, c.DataDir, poolName)
	outDir := filepath.Join(c.Root, "runs", c.Run, name)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	if c.NoText {
		defer func() {
			if serr := Scrub(outDir, filepath.Join(testDir, "items.jsonl")); serr != nil && err == nil {
				err = serr
			}
			// the pool's text must not remain either
			if serr := ScanFiles(textFiles(outDir), filepath.Join(poolDir, "items.jsonl")); serr != nil && err == nil {
				err = serr
			}
		}()
	}
	q, err := ds.ReadQuestion(filepath.Join(testDir, "question.json"))
	if err != nil {
		return err
	}
	pq, err := ds.ReadQuestion(filepath.Join(poolDir, "question.json"))
	if err != nil {
		return fmt.Errorf("training pool %s: %w", poolName, err)
	}
	if string(pq.Question) != string(q.Question) || pq.Type != q.Type || strings.Join(pq.Options, "|") != strings.Join(q.Options, "|") {
		return fmt.Errorf("training pool %s asks a different question than %s", poolName, name)
	}
	n, err := BuildStream(testDir, outDir, c.Lead, 0) // the held-out stream: items + the lead's answers (replayed for forwards)
	if err != nil {
		return err
	}
	if err := copyFile(filepath.Join(testDir, "question.json"), filepath.Join(outDir, "question.json")); err != nil {
		return err
	}
	stream, _ := ds.ReadJSONL[ds.StreamItem](filepath.Join(outDir, "stream.jsonl"))
	items, _ := ds.ReadJSONL[ds.Item](filepath.Join(testDir, "items.jsonl"))
	truth := map[string]int{}
	for _, it := range items {
		truth[it.ID] = it.Truth
	}
	leadLabel := map[string]int{}
	var op []OperateLine
	for _, s := range stream {
		leadLabel[s.ID] = ds.Argmax(s.LeadProbs)
		op = append(op, OperateLine{s.ID, s.State})
	}
	if err := ds.WriteJSONL(filepath.Join(outDir, "operate.jsonl"), op); err != nil {
		return err
	}
	rawPool, poolItems, err := labeledPrefix(poolDir, c.Lead)
	if err != nil {
		return err
	}
	pool, excluded, err := separatePool(rawPool, items)
	if err != nil {
		return fmt.Errorf("training pool %s vs test %s: %w", poolName, name, err)
	}
	res := LadderResult{Dataset: name, Pool: poolName, Lead: c.Lead, Ladder: c.Ladder, PoolReady: len(pool), PoolRawReady: len(rawPool), PoolItems: poolItems,
		PoolExcluded: excluded, PoolExcludedItems: len(rawPool) - len(pool), TestItems: n, Bar: LadderBar, Time: time.Now().UTC().Format(time.RFC3339)}
	c.Log("%s: ladder %v from pool %s (%d of %d items lead-labeled, %d excluded as near-duplicates of test prompts), held-out %d items", name, c.Ladder, poolName, len(rawPool), poolItems, len(rawPool)-len(pool), n)
	q2 := filepath.Join(outDir, "question.json")
	stream2 := filepath.Join(outDir, "stream.jsonl")
	for _, sys := range c.Systems {
		for _, rungN := range c.Ladder {
			r := Rung{System: sys, N: rungN, Bar: LadderBar}
			switch {
			case !strings.HasPrefix(sys, "understudent-"):
				r.Skipped = "the ladder needs a frozen held-out judge, which only understudent systems have (hunch learns from every request)"
			case rungN > len(pool):
				r.Skipped = fmt.Sprintf("the pool has only %d usable lead-labeled items (of %d; %d excluded as near-duplicates of test prompts)", len(pool), poolItems, len(rawPool)-len(pool))
			case strings.HasPrefix(sys, "understudent-") && !fileExists(c.UnderstudyBin):
				r.Skipped = "no understudy binary at " + c.UnderstudyBin
			}
			if r.Skipped != "" {
				c.Log("%s %s n=%d: skipped (%s)", name, sys, rungN, r.Skipped)
				res.Rungs = append(res.Rungs, r)
				continue
			}
			c.runRung(ctx, &r, sys, outDir, q2, stream2, stream, yesIndex(q.Options), pool[:rungN], leadLabel, truth, q.K())
			res.Rungs = append(res.Rungs, r)
			switch {
			case r.Error != "":
				c.Log("%s %s n=%d: FAILED: %s", name, sys, rungN, r.Error)
			default:
				c.Log("%s %s n=%d: cv %s, held-out agreement %s (majority %.3f), coverage %.3f", name, sys, rungN, pf(r.CVAgreement), pf(r.HeldAgreement), r.HeldMajority, r.Coverage)
			}
			if err := ds.WriteJSON(filepath.Join(outDir, "ladder.json"), res); err != nil {
				return err
			}
		}
	}
	return ds.WriteJSON(filepath.Join(outDir, "ladder.json"), res)
}

func pf(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("%.3f", *x)
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// textFiles lists the files of a text-free run directory other than decisions and ids (the ones ScanFiles checks).
func textFiles(outDir string) []string {
	ents, _ := os.ReadDir(outDir)
	var out []string
	for _, e := range ents {
		n := e.Name()
		if e.IsDir() || strings.HasPrefix(n, "decisions.") || n == StreamIDsFile {
			continue
		}
		out = append(out, filepath.Join(outDir, n))
	}
	return out
}

func (c Config) runRung(ctx context.Context, r *Rung, sys, outDir, q, stream string, streamItems []ds.StreamItem, yes int, train []TrainLine, leadLabel, truth map[string]int, k int) {
	understudent := strings.TrimPrefix(sys, "understudent-")
	name := LadderName(sys, r.N)
	trainFile := filepath.Join(outDir, "train.jsonl")
	if err := ds.WriteJSONL(trainFile, train); err != nil {
		r.Error = err.Error()
		return
	}
	counts := make([]int, k)
	groups := map[string]bool{}
	for _, t := range train {
		counts[ds.Argmax(t.LeadProbs)]++
		p, _ := splitParent(t.ID)
		groups[p] = true
	}
	best := 0
	for _, x := range counts {
		best = max(best, x)
	}
	r.TrainItems, r.TrainGroups, r.TrainMajority = len(train), len(groups), float64(best)/float64(len(train))
	distinct := map[string]bool{}
	for _, t := range train {
		raw, _ := json.Marshal(t.State)
		id, err := CanonicalState(raw)
		if err != nil { // not parseable: fall back to the raw bytes so the item still counts as itself
			id = "raw:" + string(raw)
		}
		distinct[id] = true
	}
	r.DedupeDrops = len(train) - len(distinct)
	snapshot := filepath.Join(outDir, "snapshot."+name)
	os.RemoveAll(snapshot)
	trainJSON := filepath.Join(outDir, "train."+name+".json")
	out := filepath.Join(outDir, "decisions."+name+".jsonl")
	os.Remove(out)
	logf, err := os.Create(filepath.Join(outDir, name+".stderr.log"))
	if err != nil {
		r.Error = err.Error()
		return
	}
	defer logf.Close()
	extra := c.UnderstudyExtra()
	steps := []step{
		{args: append([]string{c.UnderstudyBin, "train", "--understudent", understudent, "--snapshot", snapshot, "--question", q, "--examples", trainFile,
			"--max-examples", fmt.Sprint(len(train))}, extra...), stdout: trainJSON},
		{args: append([]string{c.UnderstudyBin, "judge", "--frozen", "--understudent", understudent, "--snapshot", snapshot, "--replay-lead", stream}, extra...),
			stdin: filepath.Join(outDir, "operate.jsonl"), stdout: out},
	}
	for i, s := range steps {
		st := time.Now()
		_, stderr, err := execStep(ctx, s, logf, c.Root)
		w := time.Since(st).Seconds()
		if i == 0 {
			r.TrainWallS = w
		} else {
			r.JudgeWallS = w
			if sum := LastJSONObject(stderr); sum != nil {
				ds.WriteFileAtomic(filepath.Join(outDir, "summary."+name+".json"), append(sum, '\n'))
			}
		}
		if err != nil {
			r.Error = err.Error()
			return
		}
	}
	var to trainOut
	if err := ds.ReadJSON(trainJSON, &to); err != nil {
		r.Error = "unreadable train output: " + err.Error()
		return
	}
	r.CVAgreement, r.CVCoverage, r.CVMinority, r.Mode, r.Readout, r.WhyNot = to.CVAgreement, to.CVCoverage, to.CVMinority, to.Mode, to.Readout, to.WhyNot
	// The fit must have used the whole prefix: `train` keeps only the newest --max-examples sampled examples, so a short
	// count means the oldest items were evicted and the rung is not "the first N pool items".
	r.FitN = to.N
	if to.N < len(train)-r.DedupeDrops {
		r.Error = fmt.Sprintf("the fit used %d examples, fewer than the %d pool items minus %d duplicate states: the rung does not fit the first %d items", to.N, len(train), r.DedupeDrops, r.N)
		return
	}
	if to.CVAgreement != nil && to.N > 0 {
		lo, _ := Wilson(int(math.Round(*to.CVAgreement*float64(to.N))), to.N)
		r.CVWilsonLow = fptr(lo)
	}
	dec, err := ds.ReadDecisions(out)
	if err != nil {
		r.Error = "unreadable decisions: " + err.Error()
		return
	}
	if len(dec) != len(leadLabel) {
		r.Error = fmt.Sprintf("wrote %d decisions for %d held-out items", len(dec), len(leadLabel))
		return
	}
	ScoreHeldOut(r, dec, leadLabel, truth, k)
	if tp := testPoints(streamItems, dec, yes); tp != nil {
		if pp := poolPoints(train, yes); pp != nil {
			r.LengthShortcut = ComputeLengthShortcut(pp, tp, yes)
		}
	}
	os.RemoveAll(snapshot)
	os.Remove(trainFile)
}

// LadderJSON reads runs/<run>/<ds>/ladder.json.
func LadderJSON(runDir, name string) (*LadderResult, error) {
	var l LadderResult
	if err := ds.ReadJSON(filepath.Join(runDir, name, "ladder.json"), &l); err != nil {
		return nil, err
	}
	return &l, nil
}
