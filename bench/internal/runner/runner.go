// Package runner builds runs/<run>/<ds>/stream.jsonl from items and lead answers (no truth) and runs the replay
// systems over it: understudy (Rust binary) and hunch (bench/hunch_runner, the real hunch.Proxy).
//
// Modes:
//
//	replay  cold start: each system replays the whole stream (decisions.<system>.jsonl).
//	split   warm start: the first N stream items (with lead answers) seed the understudent via train, then the rest
//	        (id + state only) is judged with the lead replayed for forwards and audits
//	        (decisions.<system>-split.jsonl, covering only the operate items).
package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/endgame-labs/understudy/bench/internal/ds"
)

// Config is one `understudy-bench run`.
type Config struct {
	Root     string
	DataDir  string // datasets directory, relative to Root unless absolute (default "data")
	NoText   bool   // after the replay delete stream/train/operate/snapshot files and verify no state text remains
	Run      string
	Lead     string
	Datasets []string
	Systems  []string // understudent, hunch
	Modes    []string // replay, split
	Train    int      // split: seeded items
	// Ladder, when set, makes the run a fixed-test learning curve (ladder.go): for each rung N, train on the first N
	// lead-labeled items of the dataset's pool (Pools[name], default DefaultPool) and score the dataset itself, frozen.
	Ladder         []int
	Pools          map[string]string
	UnderstudyBin  string
	UnderstudyArgs []string // extra args for `understudy replay` / `understudy judge`
	// EncodeCache is understudy's on-disk encode cache directory (relative to Root unless absolute), shared by every understudent
	// system and run that names it: texts are encoded once. Decisions are identical with and without it.
	EncodeCache string
	HunchArgs   []string // extra args for run_hunch.py
	Parallel    int
	Log         func(string, ...any)
}

// RunMeta is runs/<run>/run.json.
type RunMeta struct {
	Run      string   `json:"run"`
	Lead     string   `json:"lead"`
	Datasets []string `json:"datasets"`
	Systems  []string `json:"systems"`
	Modes    []string `json:"modes"`
	Train    int      `json:"train"`
	// Ladder and Pools record a fixed-test learning-curve run (--train-ladder, --train-pool).
	Ladder  []int             `json:"ladder,omitempty"`
	Pools   map[string]string `json:"pools,omitempty"`
	Time    string            `json:"time"`
	DataDir string            `json:"data_dir,omitempty"`
	// EncodeCache records that understudy used an encode cache (identical results; only speed differs).
	EncodeCache bool `json:"encode_cache,omitempty"`
	NoText      bool `json:"no_text_artifacts,omitempty"`
}

// Timing is runs/<run>/<ds>/timing.<system>.json.
type Timing struct {
	System     string   `json:"system"`
	Mode       string   `json:"mode"`
	Items      int      `json:"items"`
	WallS      float64  `json:"wall_s"`
	TrainWallS *float64 `json:"train_wall_s,omitempty"`
	ExitCode   int      `json:"exit_code"`
	Skipped    string   `json:"skipped,omitempty"`
	Error      string   `json:"error,omitempty"`
	Commands   []string `json:"commands"`
	Started    string   `json:"started"`
	Finished   string   `json:"finished"`
}

// UnderstudyExtra is the extra argument list for `understudy replay` / `understudy judge`: UnderstudyArgs, then --encode-cache.
func (c Config) UnderstudyExtra() []string {
	args := append([]string{}, c.UnderstudyArgs...)
	if c.EncodeCache != "" {
		dir := c.EncodeCache
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(c.Root, dir)
		}
		args = append(args, "--encode-cache", dir)
	}
	return args
}

// Understudents understudent accepts with --understudent.
var Understudents = []string{"deep", "probe", "keywords", "lite", "neighbors", "shape", "blend", "memo", "net", "checklist", "ladder", "auto"}

// DefaultSystems is the default --systems. "hunch" is an optional external baseline (bench/hunch_runner): ask for it with --systems.
var DefaultSystems = []string{"understudent:probe", "understudent:deep", "understudent:keywords", "understudent:lite", "understudent:neighbors", "understudent:ladder"}

// NormalizeSystem turns a --systems entry into its file name: "hunch", or "understudent:<understudent>" / "understudent" (= deep)
// into "understudent-<understudent>".
func NormalizeSystem(spec string) (string, error) {
	if spec == "hunch" {
		return spec, nil
	}
	understudent := ""
	switch {
	case spec == "understudent":
		understudent = "deep"
	case strings.HasPrefix(spec, "understudent:"):
		understudent = strings.TrimPrefix(spec, "understudent:")
	case strings.HasPrefix(spec, "understudent-"):
		understudent = strings.TrimPrefix(spec, "understudent-")
	default:
		return "", fmt.Errorf("unknown system %q (hunch, understudent, understudent:<understudent>)", spec)
	}
	for _, st := range Understudents {
		if strings.EqualFold(st, understudent) {
			return "understudent-" + st, nil
		}
	}
	return "", fmt.Errorf("unknown understudent %q (%s)", understudent, strings.Join(Understudents, ", "))
}

// SystemName is the decisions-file suffix for a system in a mode.
func SystemName(system, mode string) string {
	if mode == "split" {
		return system + "-split"
	}
	return system
}

// TrainLine is one line of train.jsonl (split mode).
type TrainLine struct {
	ID        string    `json:"id"`
	State     any       `json:"state"`
	LeadProbs []float64 `json:"lead_probs"`
}

// OperateLine is one line of operate.jsonl (split mode).
type OperateLine struct {
	ID    string `json:"id"`
	State any    `json:"state"`
}

// BuildStream writes stream.jsonl (and, with train > 0, train.jsonl and operate.jsonl) for one dataset.
func BuildStream(dataDir, outDir, lead string, train int) (int, error) {
	items, err := ds.ReadJSONL[ds.Item](filepath.Join(dataDir, "items.jsonl"))
	if err != nil {
		return 0, err
	}
	recs, err := ds.ReadJSONL[ds.LeadRecord](ds.LeadPath(dataDir, lead))
	if err != nil {
		return 0, err
	}
	by := map[string]ds.LeadRecord{}
	for _, r := range recs {
		by[r.ID] = r
	}
	stream := make([]ds.StreamItem, 0, len(items))
	var missing []string
	for _, it := range items {
		r, ok := by[it.ID]
		if ok && r.StateSHA != "" && r.StateSHA != ds.StateSHA(it.State) {
			ok = false // an answer to an older version of this item
		}
		if !ok {
			missing = append(missing, it.ID)
			continue
		}
		stream = append(stream, ds.StreamItem{ID: it.ID, State: it.State, LeadProbs: r.Probs})
	}
	if len(missing) > 0 {
		return 0, fmt.Errorf("%s: %d items have no %s answer (e.g. %s): run `understudy-bench teach` first", dataDir,
			len(missing), lead, missing[0])
	}
	if err := ds.WriteJSONL(filepath.Join(outDir, "stream.jsonl"), stream); err != nil {
		return 0, err
	}
	if train > 0 {
		if train >= len(stream) {
			return 0, fmt.Errorf("--train %d leaves nothing to operate on (%d items)", train, len(stream))
		}
		var tr []TrainLine
		var op []OperateLine
		for i, s := range stream {
			if i < train {
				tr = append(tr, TrainLine{s.ID, s.State, s.LeadProbs})
			} else {
				op = append(op, OperateLine{s.ID, s.State})
			}
		}
		if err := ds.WriteJSONL(filepath.Join(outDir, "train.jsonl"), tr); err != nil {
			return 0, err
		}
		if err := ds.WriteJSONL(filepath.Join(outDir, "operate.jsonl"), op); err != nil {
			return 0, err
		}
	}
	return len(stream), nil
}

// Run executes the configured run.
func Run(ctx context.Context, c Config) error {
	runDir := filepath.Join(c.Root, "runs", c.Run)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return err
	}
	meta := RunMeta{Run: c.Run, Lead: c.Lead, Datasets: c.Datasets, Systems: c.Systems, Modes: c.Modes,
		Train: c.Train, Ladder: c.Ladder, Pools: c.Pools, Time: time.Now().UTC().Format(time.RFC3339), DataDir: c.DataDir, NoText: c.NoText, EncodeCache: c.EncodeCache != ""}
	// merge with an earlier run.json so a run can be extended dataset by dataset
	var old RunMeta
	if ds.ReadJSON(filepath.Join(runDir, "run.json"), &old) == nil {
		meta.Datasets = union(old.Datasets, meta.Datasets)
		meta.Systems = union(old.Systems, meta.Systems)
		meta.Modes = union(old.Modes, meta.Modes)
		if old.Lead != "" && old.Lead != c.Lead {
			return fmt.Errorf("run %s was built with lead %s, not %s", c.Run, old.Lead, c.Lead)
		}
		if normData(old.DataDir) != normData(meta.DataDir) {
			return fmt.Errorf("run %s was built from data dir %q, not %q", c.Run, old.DataDir, meta.DataDir)
		}
		meta.NoText = meta.NoText || old.NoText
	}
	if err := ds.WriteJSON(filepath.Join(runDir, "run.json"), meta); err != nil {
		return err
	}
	par := max(1, c.Parallel)
	sem := make(chan struct{}, par)
	errs := make(chan error, len(c.Datasets))
	for _, name := range c.Datasets {
		sem <- struct{}{}
		go func() {
			defer func() { <-sem }()
			errs <- c.dataset(ctx, name)
		}()
	}
	for i := 0; i < par; i++ {
		sem <- struct{}{}
	}
	close(errs)
	var all []string
	for e := range errs {
		if e != nil {
			all = append(all, e.Error())
		}
	}
	if len(all) > 0 {
		return fmt.Errorf("%s", strings.Join(all, "\n"))
	}
	return nil
}

func normData(d string) string {
	if d == "" {
		return "data"
	}
	return d
}

func union(a, b []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, x := range append(append([]string{}, a...), b...) {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}

func (c Config) dataset(ctx context.Context, name string) (err error) {
	dataDir := DataPath(c.Root, c.DataDir, name)
	outDir := filepath.Join(c.Root, "runs", c.Run, name)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	if c.NoText {
		// also on failure or interruption: the stream holds benchmark text that must not be left in runs/
		defer func() {
			if serr := Scrub(outDir, filepath.Join(dataDir, "items.jsonl")); serr != nil && err == nil {
				err = serr
			}
		}()
	}
	if len(c.Ladder) > 0 {
		return c.ladderDataset(ctx, name)
	}
	train := 0
	if contains(c.Modes, "split") {
		train = c.Train
	}
	n, err := BuildStream(dataDir, outDir, c.Lead, train)
	if err != nil {
		return err
	}
	if err := copyFile(filepath.Join(dataDir, "question.json"), filepath.Join(outDir, "question.json")); err != nil {
		return err
	}
	c.Log("%s: stream of %d items", name, n)
	for _, mode := range c.Modes {
		for _, sys := range c.Systems {
			t := c.runSystem(ctx, sys, mode, outDir, n)
			if err := ds.WriteJSON(filepath.Join(outDir, "timing."+SystemName(sys, mode)+".json"), t); err != nil {
				return err
			}
			switch {
			case t.Skipped != "":
				c.Log("%s %s: skipped (%s)", name, SystemName(sys, mode), t.Skipped)
			case t.Error != "":
				c.Log("%s %s: FAILED after %.1fs: %s (see %s.stderr.log)", name, SystemName(sys, mode), t.WallS, t.Error, SystemName(sys, mode))
			default:
				c.Log("%s %s: %.1fs", name, SystemName(sys, mode), t.WallS)
			}
		}
	}
	return nil
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

func copyFile(src, dst string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return ds.WriteFileAtomic(dst, b)
}

// step is one process to run: args, optional stdin file and stdout file.
type step struct {
	args          []string
	stdin, stdout string
}

func (s step) String() string {
	cmd := strings.Join(s.args, " ")
	if s.stdin != "" {
		cmd += " < " + s.stdin
	}
	if s.stdout != "" {
		cmd += " > " + s.stdout
	}
	return cmd
}

func (c Config) runSystem(ctx context.Context, sys, mode, outDir string, n int) Timing {
	name := SystemName(sys, mode)
	t := Timing{System: sys, Mode: mode, Items: n, Started: time.Now().UTC().Format(time.RFC3339)}
	q := filepath.Join(outDir, "question.json")
	stream := filepath.Join(outDir, "stream.jsonl")
	out := filepath.Join(outDir, "decisions."+name+".jsonl")
	os.Remove(out)
	var steps []step
	kind := sys
	if strings.HasPrefix(sys, "understudent-") {
		kind = "understudent"
	}
	switch kind {
	case "understudent":
		understudent := strings.TrimPrefix(sys, "understudent-")
		if _, err := os.Stat(c.UnderstudyBin); err != nil {
			t.Skipped = "no understudy binary at " + c.UnderstudyBin
			return t
		}
		if mode == "replay" {
			steps = []step{{args: append([]string{c.UnderstudyBin, "replay", "--understudent", understudent, "--question", q, "--stream", stream, "--out", out}, c.UnderstudyExtra()...)}}
		} else {
			snapshot := filepath.Join(outDir, "snapshot."+sys)
			os.RemoveAll(snapshot)
			steps = []step{
				{args: []string{c.UnderstudyBin, "train", "--understudent", understudent, "--snapshot", snapshot, "--question", q, "--examples", filepath.Join(outDir, "train.jsonl")}},
				{args: append([]string{c.UnderstudyBin, "judge", "--understudent", understudent, "--snapshot", snapshot, "--replay-lead", stream}, c.UnderstudyExtra()...),
					stdin: filepath.Join(outDir, "operate.jsonl"), stdout: out},
			}
		}
	case "hunch":
		project := filepath.Join(c.Root, "bench", "hunch_runner")
		args := []string{"uv", "run", "--project", project, "python", filepath.Join(project, "run_hunch.py"),
			"--question", q, "--stream", stream, "--out", out}
		if mode == "split" {
			args = append(args, "--train-examples", filepath.Join(outDir, "train.jsonl"), "--operate", filepath.Join(outDir, "operate.jsonl"))
		}
		steps = []step{{args: append(args, c.HunchArgs...)}}
	default:
		t.Error = "unknown system " + sys
		return t
	}
	logf, err := os.Create(filepath.Join(outDir, name+".stderr.log"))
	if err != nil {
		t.Error = err.Error()
		return t
	}
	defer logf.Close()
	t0 := time.Now()
	for i, s := range steps {
		t.Commands = append(t.Commands, s.String())
		st := time.Now()
		code, stderr, err := execStep(ctx, s, logf, c.Root)
		if sum := LastJSONObject(stderr); sum != nil {
			path := filepath.Join(outDir, "summary."+name+".json")
			if i < len(steps)-1 {
				path = filepath.Join(outDir, "summary-train."+name+".json")
			}
			ds.WriteFileAtomic(path, append(sum, '\n'))
		}
		if i == 0 && len(steps) > 1 {
			w := time.Since(st).Seconds()
			t.TrainWallS = &w
		}
		if err != nil {
			t.ExitCode, t.Error = code, err.Error()
			break
		}
	}
	t.WallS = time.Since(t0).Seconds()
	if t.Error == "" {
		want := n
		if mode == "split" {
			want = n - c.Train
		}
		if got, err := ds.ReadDecisions(out); err != nil {
			t.Error = "unreadable decisions: " + err.Error()
		} else if len(got) != want {
			t.Error = fmt.Sprintf("wrote %d decisions for %d items", len(got), want)
		}
	}
	t.Finished = time.Now().UTC().Format(time.RFC3339)
	return t
}

func execStep(ctx context.Context, s step, logf io.Writer, dir string) (int, []byte, error) {
	cmd := exec.CommandContext(ctx, s.args[0], s.args[1:]...)
	cmd.Dir = dir
	fmt.Fprintf(logf, "$ %s\n", s)
	var all bytes.Buffer
	cmd.Stderr = io.MultiWriter(logf, &all)
	if s.stdin != "" {
		f, err := os.Open(s.stdin)
		if err != nil {
			return -1, nil, err
		}
		defer f.Close()
		cmd.Stdin = f
	}
	if s.stdout != "" {
		f, err := os.Create(s.stdout)
		if err != nil {
			return -1, nil, err
		}
		defer f.Close()
		cmd.Stdout = f
	} else {
		cmd.Stdout = logf
	}
	err := cmd.Run()
	if err != nil {
		code := -1
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		}
		lines := strings.Split(strings.TrimSpace(all.String()), "\n")
		return code, all.Bytes(), fmt.Errorf("%v: %s", err, lines[len(lines)-1])
	}
	return 0, all.Bytes(), nil
}

// LastJSONObject returns the last JSON object in a process's stderr: either a one-line object or a pretty-printed
// one starting with a "{" line. nil when there is none.
func LastJSONObject(b []byte) []byte {
	text := strings.TrimRight(string(b), "\n\r\t ")
	if !strings.HasSuffix(text, "}") {
		return nil
	}
	for i := len(text) - 1; i >= 0; i-- {
		if text[i] == '{' && (i == 0 || text[i-1] == '\n') {
			cand := []byte(text[i:])
			if json.Valid(cand) {
				var buf bytes.Buffer
				if json.Compact(&buf, cand) == nil {
					return buf.Bytes()
				}
				return cand
			}
		}
	}
	return nil
}
