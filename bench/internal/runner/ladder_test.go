package runner

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dorkitude/understudy/bench/internal/ds"
)

func TestParseLadderAndPool(t *testing.T) {
	got, err := ParseLadder([]string{"200,1000", "2000"})
	if err != nil || fmt.Sprint(got) != "[200 1000 2000]" {
		t.Fatal(got, err)
	}
	for _, bad := range [][]string{{"200,100"}, {"0"}, {"x"}, {""}, {"1e3"}, {"200,200"}} {
		if _, err := ParseLadder(bad); err == nil {
			t.Errorf("accepted %v", bad)
		}
	}
	if DefaultPool("judge_rb2_safety_per") != "judge_rb2_safety_big_per" || DefaultPool("x") != "x_big" {
		t.Fatal("DefaultPool")
	}
	if lo, hi := Wilson(27, 30); math.Abs(lo-0.7438) > 1e-4 || math.Abs(hi-0.9654) > 1e-4 {
		t.Fatal(lo, hi)
	}
	if lo, hi := Wilson(0, 0); lo != 0 || hi != 1 {
		t.Fatal(lo, hi)
	}
}

func TestScoreHeldOut(t *testing.T) {
	one, zero := 1, 0
	lead := map[string]int{"d-0001#1": 1, "d-0001#2": 0, "d-0002#1": 0, "d-0002#2": 0}
	truth := map[string]int{"d-0001#1": 1, "d-0001#2": 0, "d-0002#1": 1, "d-0002#2": 0}
	dec := []ds.Decision{
		{ID: "d-0001#1", AnsweredBy: "understudent", Label: 1, UnderstudentLabel: &one},
		{ID: "d-0001#2", AnsweredBy: "lead", Label: 0, UnderstudentLabel: &zero},
		{ID: "d-0002#1", AnsweredBy: "lead", Label: 0, UnderstudentLabel: &one}, // understudent wrong
		{ID: "d-0002#2", AnsweredBy: "lead", Label: 0, UnderstudentLabel: nil},  // no prediction: not counted
	}
	var r Rung
	ScoreHeldOut(&r, dec, lead, truth, 2)
	if r.HeldN != 4 || r.HeldMajority != 0.75 || r.HeldGroupsApprox != 2 {
		t.Fatalf("%+v", r)
	}
	if *r.HeldAgreement != 2.0/3 || r.Coverage != 0.25 || r.LocalN != 1 || *r.LocalAgreement != 1 || r.ClearsBar || r.LocalClearsBar {
		t.Fatalf("%+v", r)
	}
	if got := *r.HeldAccTruth; got != 1 {
		t.Fatalf("acc vs truth %v", got)
	}
}

// fakeUnderstudy is a stand-in for the understudy binary: it logs its arguments and the number of training lines,
// prints a train summary, and answers every judged line with understudent_label = 1 (and, past the first line, locally).
const fakeUnderstudy = `#!/bin/sh
echo "$@" >> "$FAKE_LOG"
case "$1" in
train)
  while [ $# -gt 0 ]; do [ "$1" = "--examples" ] && ex="$2"; [ "$1" = "--max-examples" ] && cap="$2"; shift; done
  lines=$(wc -l < "$ex")
  echo "train lines: $lines" >> "$FAKE_LOG"
  echo "max-examples: $cap" >> "$FAKE_LOG"
  # the fit keeps one example per distinct state (canonical, like state_text: key order does not matter), and at most the cap (the real train evicts the oldest past it)
  n=$(sed 's/.*"state"://; s/,"lead_probs".*//' "$ex" | python3 -I -c 'import sys,json; [print(json.dumps(json.loads(l),sort_keys=True)) for l in sys.stdin]' | sort -u | wc -l)
  [ -n "$cap" ] && [ "$n" -gt "$cap" ] && n=$cap
  n=$((n - ${FAKE_SHORT:-0}))
  echo '{"n": '$n', "cv_agreement": 0.8, "cv_minority_recall": 0.5, "cv_coverage": 0.0, "mode": null, "readout": "mean", "why_not": "toy"}'
  ;;
judge)
  i=0
  while read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
    i=$((i+1))
    by=lead; [ $i -gt 1 ] && by=understudent
    echo "{\"id\":\"$id\",\"answered_by\":\"$by\",\"label\":1,\"probs\":[0.1,0.9],\"understudent_label\":1,\"understudent_probs\":[0.1,0.9],\"eligible\":true,\"novel\":false,\"demoted\":false,\"mode\":\"symmetric\",\"audited\":false,\"readout\":\"mean\",\"understudent_us\":1}"
  done
  echo '{"items": 8}' >&2
  ;;
esac
`

// uniquePrompt is an instruction of eight words unique to the key, so no two toy prompts are near-duplicates.
func uniquePrompt(key string) string {
	h := uint64(14695981039346656037)
	for _, c := range []byte(key) {
		h = (h ^ uint64(c)) * 1099511628211
	}
	var w []string
	for k := 0; k < 8; k++ {
		h = h*6364136223846793005 + 1442695040888963407
		x := h >> 20
		var b []byte
		for i := 0; i < 6; i++ {
			b = append(b, byte('a'+x%26))
			x /= 26
		}
		w = append(w, string(b))
	}
	return "Please " + strings.Join(w, " ")
}

func writeToy(t *testing.T, dir, name string, ids []string, probs [][]float64, truth []int) {
	t.Helper()
	d := filepath.Join(dir, "data-private", name)
	os.MkdirAll(d, 0o755)
	q := `{"name":"` + name + `","category":"real_judging","type":"noul","options":["false","true"],"question":{"type":"noul","instructions":"Is it best?","criteria":{"true":"y","false":"n"}},"pair":["instruction","response"],"notes":""}`
	os.WriteFile(filepath.Join(d, "question.json"), []byte(q), 0o644)
	var items []ds.Item
	var recs []ds.LeadRecord
	for i, id := range ids {
		parent, _ := splitParent(id)
		st := []byte(fmt.Sprintf(`{"instruction":"%s","response":"response number %d"}`, uniquePrompt(parent), i))
		items = append(items, ds.Item{ID: id, State: st, Truth: truth[i]})
		recs = append(recs, ds.LeadRecord{ID: id, Probs: probs[i], Label: ds.Argmax(probs[i]), Model: "m", StateSHA: ds.StateSHA(st)})
	}
	ds.WriteJSONL(filepath.Join(d, "items.jsonl"), items)
	ds.WriteJSONL(filepath.Join(d, "lead.jev.jsonl"), recs)
}

func TestLadderTrainsOnPrefixAndScoresFixedTest(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "SPEC.md"), nil, 0o644)
	yes, no := []float64{0.2, 0.8}, []float64{0.8, 0.2}
	// held-out test: 2 prompts x 4 responses; pool: 3 prompts x 4 responses, but only the first 2 prompts lead-labeled
	var tids, pids []string
	var tp, pp [][]float64
	var tt, pt []int
	for p := 1; p <= 2; p++ {
		for k := 1; k <= 4; k++ {
			tids = append(tids, fmt.Sprintf("t_per-%04d#%d", p, k))
			tp, tt = append(tp, map[bool][]float64{true: yes, false: no}[k == 1]), append(tt, map[bool]int{true: 1, false: 0}[k == 1])
		}
	}
	for p := 1; p <= 3; p++ {
		for k := 1; k <= 4; k++ {
			pids = append(pids, fmt.Sprintf("t_big_per-%04d#%d", p, k))
			pp, pt = append(pp, no), append(pt, 0)
		}
	}
	writeToy(t, root, "t_per", tids, tp, tt)
	writeToy(t, root, "t_big_per", pids, pp, pt)
	// only 8 pool items are lead-labeled: drop the third prompt's answers
	recs, _ := ds.ReadJSONL[ds.LeadRecord](filepath.Join(root, "data-private", "t_big_per", "lead.jev.jsonl"))
	ds.WriteJSONL(filepath.Join(root, "data-private", "t_big_per", "lead.jev.jsonl"), recs[:8])

	bin := filepath.Join(root, "fake-understudy")
	os.WriteFile(bin, []byte(fakeUnderstudy), 0o755)
	log := filepath.Join(root, "fake.log")
	t.Setenv("FAKE_LOG", log)
	c := Config{Root: root, DataDir: "data-private", NoText: true, Run: "lad", Lead: "jev", Datasets: []string{"t_per"},
		Systems: []string{"understudent-deep", "hunch"}, Modes: []string{"ladder"}, Ladder: []int{4, 8, 12}, UnderstudyBin: bin, Parallel: 1,
		Log: func(string, ...any) {}}
	if err := Run(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	l, err := LadderJSON(filepath.Join(root, "runs", "lad"), "t_per")
	if err != nil {
		t.Fatal(err)
	}
	if l.Pool != "t_big_per" || l.PoolReady != 8 || l.PoolItems != 12 || l.TestItems != 8 {
		t.Fatalf("%+v", l)
	}
	if len(l.Rungs) != 6 {
		t.Fatalf("%d rungs", len(l.Rungs))
	}
	by := map[string]Rung{}
	for _, r := range l.Rungs {
		by[LadderName(r.System, r.N)] = r
	}
	if r := by["hunch-n4"]; !strings.Contains(r.Skipped, "frozen") {
		t.Fatalf("hunch should be skipped: %+v", r)
	}
	if r := by["understudent-deep-n12"]; !strings.Contains(r.Skipped, "only 8 usable lead-labeled") {
		t.Fatalf("n=12 should be skipped (pool not fully labeled): %+v", r)
	}
	r4, r8 := by["understudent-deep-n4"], by["understudent-deep-n8"]
	if r4.Error != "" || r8.Error != "" || r4.TrainItems != 4 || r8.TrainItems != 8 || r4.TrainGroups != 1 || r8.TrainGroups != 2 {
		t.Fatalf("%+v\n%+v", r4, r8)
	}
	// the fake labels everything 1; the lead says 1 for slot 1 of each prompt only: agreement 2/8, majority 6/8
	if *r4.HeldAgreement != 0.25 || r4.HeldMajority != 0.75 || *r4.HeldLift != -0.5 || r4.ClearsBar {
		t.Fatalf("%+v", r4)
	}
	if r4.Coverage != 7.0/8 || r4.LocalN != 7 || *r4.CVAgreement != 0.8 || r4.CVWilsonLow == nil || r4.Mode != nil {
		t.Fatalf("%+v", r4)
	}
	if r4.FitN != 4 || r8.FitN != 8 || r4.DedupeDrops != 0 || l.PoolRawReady != 8 || len(l.PoolExcluded) != 0 {
		t.Fatalf("%+v %+v %+v", r4, r8, l)
	}
	if want, _ := Wilson(int(math.Round(0.8*4)), 4); *r4.CVWilsonLow != want {
		t.Fatalf("CV Wilson bound must use the fitted n: %v want %v", *r4.CVWilsonLow, want)
	}
	if r4.TrainMajority != 1 {
		t.Fatalf("pool labels are all 'no': %+v", r4)
	}
	// the length-shortcut diagnostic is on every rung: the pool prefix and the test are bucketed by the test's quartiles
	if ls := r4.LengthShortcut; ls == nil || ls.PoolN[0]+ls.PoolN[1]+ls.PoolN[2]+ls.PoolN[3] != 4 || ls.TestN[0]+ls.TestN[1]+ls.TestN[2]+ls.TestN[3] != 8 {
		t.Fatalf("length shortcut: %+v", ls)
	} else if ls.Delta != nil || ls.SpearmanUnderstudent != nil {
		t.Fatalf("the fake understudent's p(yes) is constant, so its correlation with length is undefined: %+v", ls)
	}
	if ls := r8.LengthShortcut; ls == nil || ls.PoolN[0]+ls.PoolN[1]+ls.PoolN[2]+ls.PoolN[3] != 8 {
		t.Fatalf("length shortcut at 8: %+v", ls)
	}
	logb, _ := os.ReadFile(log)
	got := string(logb)
	for _, want := range []string{"judge --frozen --understudent deep", "--replay-lead", "train lines: 4", "train lines: 8", "--max-examples 4", "--max-examples 8"} {
		if !strings.Contains(got, want) {
			t.Errorf("fake understudent log lacks %q:\n%s", want, got)
		}
	}
	// text-free: no stream/train/operate/snapshot left, no state text anywhere
	out := filepath.Join(root, "runs", "lad", "t_per")
	for _, gone := range []string{"stream.jsonl", "train.jsonl", "operate.jsonl", "snapshot.understudent-deep-n4"} {
		if _, err := os.Stat(filepath.Join(out, gone)); err == nil {
			t.Errorf("%s left behind", gone)
		}
	}
	if err := VerifyNoText(out, filepath.Join(root, "data-private", "t_big_per", "items.jsonl")); err != nil {
		t.Fatal(err)
	}
}

// ladderToy builds a held-out set (2 prompts x 4) and a fully labeled pool of poolPrompts prompts x 4 in a temp root.
func ladderToy(t *testing.T, poolPrompts int) (Config, string) {
	t.Helper()
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "SPEC.md"), nil, 0o644)
	yes, no := []float64{0.2, 0.8}, []float64{0.8, 0.2}
	var tids, pids []string
	var tp, pp [][]float64
	var tt, pt []int
	for p := 1; p <= 2; p++ {
		for k := 1; k <= 4; k++ {
			tids = append(tids, fmt.Sprintf("t_per-%04d#%d", p, k))
			tp, tt = append(tp, map[bool][]float64{true: yes, false: no}[k == 1]), append(tt, map[bool]int{true: 1, false: 0}[k == 1])
		}
	}
	for p := 1; p <= poolPrompts; p++ {
		for k := 1; k <= 4; k++ {
			pids = append(pids, fmt.Sprintf("t_big_per-%04d#%d", p, k))
			pp, pt = append(pp, no), append(pt, 0)
		}
	}
	writeToy(t, root, "t_per", tids, tp, tt)
	writeToy(t, root, "t_big_per", pids, pp, pt)
	bin := filepath.Join(root, "fake-understudy")
	os.WriteFile(bin, []byte(fakeUnderstudy), 0o755)
	t.Setenv("FAKE_LOG", filepath.Join(root, "fake.log"))
	return Config{Root: root, DataDir: "data-private", NoText: true, Run: "lad", Lead: "jev", Datasets: []string{"t_per"},
		Systems: []string{"understudent-deep"}, Modes: []string{"ladder"}, Ladder: []int{4, 8}, UnderstudyBin: bin, Parallel: 1,
		Log: func(string, ...any) {}}, root
}

// rewritePool replaces the pool's state of items (and the lead answers' hashes) through f.
func rewritePool(t *testing.T, root string, f func(i int, it *ds.Item)) {
	t.Helper()
	d := filepath.Join(root, "data-private", "t_big_per")
	items, _ := ds.ReadJSONL[ds.Item](filepath.Join(d, "items.jsonl"))
	recs, _ := ds.ReadJSONL[ds.LeadRecord](filepath.Join(d, "lead.jev.jsonl"))
	for i := range items {
		f(i, &items[i])
		recs[i].StateSHA = ds.StateSHA(items[i].State)
	}
	ds.WriteJSONL(filepath.Join(d, "items.jsonl"), items)
	ds.WriteJSONL(filepath.Join(d, "lead.jev.jsonl"), recs)
}

func ladderRung(t *testing.T, root, name string) Rung {
	t.Helper()
	l, err := LadderJSON(filepath.Join(root, "runs", "lad"), "t_per")
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range l.Rungs {
		if LadderName(r.System, r.N) == name {
			return r
		}
	}
	t.Fatalf("no rung %s", name)
	return Rung{}
}

// The ladder must fail when the fit used fewer examples than the rung's pool items (the cap evicted the oldest ones).
func TestLadderFailsWhenFitIsShort(t *testing.T) {
	c, root := ladderToy(t, 3)
	t.Setenv("FAKE_SHORT", "1")
	if err := Run(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	r := ladderRung(t, root, "understudent-deep-n4")
	if !strings.Contains(r.Error, "fewer than") || r.FitN != 3 || r.TrainItems != 4 {
		t.Fatalf("%+v", r)
	}
}

// Duplicate states in the first N are dropped by the fit: the ladder reports them and accepts the smaller count.
func TestLadderReportsDedupeDrops(t *testing.T) {
	c, root := ladderToy(t, 3)
	var first json.RawMessage
	rewritePool(t, root, func(i int, it *ds.Item) {
		if i == 0 {
			first = it.State
		}
		if i == 1 {
			it.State = first // identical state to item 1
		}
	})
	if err := Run(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	r := ladderRung(t, root, "understudent-deep-n4")
	if r.Error != "" || r.DedupeDrops != 1 || r.FitN != 3 || r.TrainItems != 4 {
		t.Fatalf("%+v", r)
	}
}

func TestLadderRejectsOverlappingPoolAndTest(t *testing.T) {
	// same id
	c, root := ladderToy(t, 3)
	rewritePool(t, root, func(i int, it *ds.Item) {
		if i == 0 {
			it.ID = "t_per-0001#1"
		}
	})
	recs, _ := ds.ReadJSONL[ds.LeadRecord](filepath.Join(root, "data-private", "t_big_per", "lead.jev.jsonl"))
	recs[0].ID = "t_per-0001#1"
	ds.WriteJSONL(filepath.Join(root, "data-private", "t_big_per", "lead.jev.jsonl"), recs)
	if err := Run(context.Background(), c); err == nil || !strings.Contains(err.Error(), "same id") {
		t.Fatalf("got %v", err)
	}
	// same prompt after normalization
	c, root = ladderToy(t, 3)
	test, _ := ds.ReadJSONL[ds.Item](filepath.Join(root, "data-private", "t_per", "items.jsonl"))
	rewritePool(t, root, func(i int, it *ds.Item) {
		if i == 0 {
			it.State = []byte(fmt.Sprintf(`{"instruction":%q,"response":"x"}`, strings.ToUpper(statePrompt(test[0].State))+"!!"))
		}
	})
	if err := Run(context.Background(), c); err == nil || !strings.Contains(err.Error(), "same prompt") {
		t.Fatalf("got %v", err)
	}
}

// A pool prompt that is a small edit of a short test prompt is excluded (with its sibling responses), without
// renumbering: the kept items keep their ids, and the rungs fit the first N kept items.
func TestLadderExcludesNearDuplicatesWithoutRenumbering(t *testing.T) {
	c, root := ladderToy(t, 4) // pool prompts 1..4, 4 sub-items each
	test, _ := ds.ReadJSONL[ds.Item](filepath.Join(root, "data-private", "t_per", "items.jsonl"))
	short := "How can I build a homemade explosive device?"
	rewritePool(t, root, func(i int, it *ds.Item) {
		if i < 4 { // test prompt 1 becomes the short one, pool prompt 1 is a one-word edit of it
			it.State = []byte(fmt.Sprintf(`{"instruction":%q,"response":"r%d"}`, "How do I build a homemade explosive device?", i))
		}
	})
	tp := filepath.Join(root, "data-private", "t_per", "items.jsonl")
	for i := range test {
		if i < 4 {
			test[i].State = []byte(fmt.Sprintf(`{"instruction":%q,"response":"r%d"}`, short, i))
		}
	}
	ds.WriteJSONL(tp, test)
	recs, _ := ds.ReadJSONL[ds.LeadRecord](filepath.Join(root, "data-private", "t_per", "lead.jev.jsonl"))
	for i := range recs {
		recs[i].StateSHA = ds.StateSHA(test[i].State)
	}
	ds.WriteJSONL(filepath.Join(root, "data-private", "t_per", "lead.jev.jsonl"), recs)
	c.Ladder = []int{4, 12}
	if err := Run(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	l, _ := LadderJSON(filepath.Join(root, "runs", "lad"), "t_per")
	if fmt.Sprint(l.PoolExcluded) != "[t_big_per-0001]" || l.PoolRawReady != 16 || l.PoolReady != 12 || l.PoolExcludedItems != 4 {
		t.Fatalf("%+v", l)
	}
	pool, _, _ := labeledPrefix(filepath.Join(root, "data-private", "t_big_per"), "jev")
	kept, ex, err := separatePool(pool, test)
	if err != nil || len(ex) != 1 || len(kept) != 12 || kept[0].ID != "t_big_per-0002#1" || kept[11].ID != "t_big_per-0004#4" {
		t.Fatalf("%v %v %d", err, ex, len(kept))
	}
	if r := ladderRung(t, root, "understudent-deep-n12"); r.Error != "" || r.TrainItems != 12 {
		t.Fatalf("%+v", r)
	}
}

func TestLadderRejectsMismatchedPoolQuestion(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "SPEC.md"), nil, 0o644)
	writeToy(t, root, "a_per", []string{"a_per-0001#1"}, [][]float64{{0.5, 0.5}}, []int{0})
	writeToy(t, root, "a_big_per", []string{"a_big_per-0001#1"}, [][]float64{{0.5, 0.5}}, []int{0})
	p := filepath.Join(root, "data-private", "a_big_per", "question.json")
	b, _ := os.ReadFile(p)
	os.WriteFile(p, []byte(strings.Replace(string(b), "Is it best?", "Another question?", 1)), 0o644)
	c := Config{Root: root, DataDir: "data-private", Run: "x", Lead: "jev", Datasets: []string{"a_per"}, Systems: []string{"understudent-deep"},
		Modes: []string{"ladder"}, Ladder: []int{1}, UnderstudyBin: "/nonexistent", Parallel: 1, Log: func(string, ...any) {}}
	if err := Run(context.Background(), c); err == nil || !strings.Contains(err.Error(), "different question") {
		t.Fatalf("got %v", err)
	}
}

// States equal up to key order or spacing are one state to the Rust fit (state_text sorts keys), so the ladder counts
// them as duplicates too.
func TestLadderDedupeUsesCanonicalState(t *testing.T) {
	c, root := ladderToy(t, 3)
	rewritePool(t, root, func(i int, it *ds.Item) {
		switch i {
		case 0:
			it.State = json.RawMessage(`{"zeta":"a b","response":"x","n":1.50}`)
		case 1:
			it.State = json.RawMessage(`{"n":1.5,"response":"x","zeta":"a b"}`) // same state, reordered
		}
	})
	if err := Run(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	r := ladderRung(t, root, "understudent-deep-n4")
	if r.Error != "" || r.DedupeDrops != 1 || r.FitN != 3 {
		t.Fatalf("%+v", r)
	}
}

func TestCanonicalStateMatchesRust(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{`"hi"`, "hi"},
		{`{"output":"ok","call":"ls","exit":0,"cwd":"/tmp"}`, "ls\nok\ncwd: /tmp\nexit: 0"},
		{`{"b":1,"a":[1,"x"]}`, `{"a": [1, "x"], "b": 1}`},
		{`{"call":null,"tool_call":"x","output":"y"}`, `{"call": null, "output": "y", "tool_call": "x"}`},
		{`{"call":{"z":true,"a":null},"output":"o"}`, "{'z': True, 'a': None}\no"},
		{`[1e-5, 1.0, 1e16, 2.5]`, `[1e-05, 1.0, 1e+16, 2.5]`},
	} {
		got, err := CanonicalState([]byte(c.in))
		if err != nil || got != c.want {
			t.Errorf("%s: got %q (%v), want %q", c.in, got, err, c.want)
		}
	}
}
