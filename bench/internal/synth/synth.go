// Package synth builds the benchmark datasets: GLM 5.3 Flash writes the surface (we assign every target label, so
// the label we asked for is the ground truth), strict validation drops malformed items, quotas fix the label
// balance, and the stream order is shuffled with a fixed seed.
package synth

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dorkitude/understudy/bench/internal/ds"
	"github.com/dorkitude/understudy/bench/internal/llm"
)

// Member is one stream item before it gets an id.
type Member struct {
	State any
	Truth int
}

// Unit is what quotas count: usually one item; a minimal pair is one unit of two members.
type Unit struct {
	Class   int
	Members []Member
}

// Dataset defines one dataset.
type Dataset struct {
	Name, Category, Type string
	Options              []string
	Question             ds.OMap
	Pair                 []string
	Notes                string
	// Build returns n stream items' worth of units (in generation order) plus extra stats.
	Build func(ctx context.Context, b *Builder, n int) ([]Unit, error)
	// NoAdjacentGroups keeps members of one unit apart in the stream.
	NoAdjacentGroups bool
}

// Builder runs one dataset's generation.
type Builder struct {
	DS    *Dataset
	LLM   *llm.Client
	Cache *llm.Cache
	Seed  uint64

	mu    sync.Mutex
	Drops map[string]int // reason -> count
	Batch map[string]int // batches by outcome
	Log   func(format string, args ...any)
}

// Drop counts a dropped item or batch.
func (b *Builder) Drop(reason string) {
	b.mu.Lock()
	b.Drops[reason]++
	b.mu.Unlock()
}

func (b *Builder) countBatch(outcome string) {
	b.mu.Lock()
	b.Batch[outcome]++
	b.mu.Unlock()
}

// RNG is a deterministic generator for (dataset, purpose, index).
func (b *Builder) RNG(purpose string, i int) *rand.Rand {
	h := fnv.New64a()
	fmt.Fprintf(h, "%s/%s/%d", b.DS.Name, purpose, i)
	return rand.New(rand.NewPCG(b.Seed, h.Sum64()))
}

// Job is one planned GLM call and how to read its reply.
type Job struct {
	Name      string // logical cache name, e.g. "tool_failed/batch/00012"
	System    string
	User      string
	MaxTokens int
	Parse     func(items []map[string]any) []Unit // invalid items are dropped (and counted via Drop)
	Planned   int                                 // units asked for
}

// Run executes jobs concurrently (bounded by the client's semaphore) and returns their units in job order.
func (b *Builder) Run(ctx context.Context, jobs []Job) [][]Unit {
	out := make([][]Unit, len(jobs))
	var wg sync.WaitGroup
	for i, j := range jobs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			content, err := b.LLM.Complete(ctx, b.Cache, j.Name, j.System, j.User, j.MaxTokens)
			if err != nil {
				b.countBatch("call_failed")
				b.Drop("batch_call_failed")
				return
			}
			items, err := llm.ParseItems(content)
			if err != nil {
				b.countBatch("malformed")
				b.Drop("batch_malformed_json")
				return
			}
			if j.Planned >= 0 && len(items) != j.Planned {
				b.Drop(fmt.Sprintf("batch_item_count_mismatch(%d of %d)", len(items), j.Planned))
			}
			b.countBatch("ok")
			out[i] = j.Parse(items)
		}()
	}
	wg.Wait()
	return out
}

// Fill plans batches with plan(i) until the quotas (units per class) are met, then selects units in batch order.
// It runs rounds: a first estimate with 15% headroom, then more batches sized by the observed yield.
func (b *Builder) Fill(ctx context.Context, quotas []int, perBatch int, plan func(i int) Job) ([]Unit, error) {
	total := 0
	for _, q := range quotas {
		total += q
	}
	var all [][]Unit
	next := 0
	batches := int(math.Ceil(float64(total)/float64(perBatch)*1.15)) + 1
	for round := 0; round < 8; round++ {
		jobs := make([]Job, 0, batches)
		for i := 0; i < batches; i++ {
			jobs = append(jobs, plan(next+i))
		}
		next += batches
		all = append(all, b.Run(ctx, jobs)...)
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		have := make([]int, len(quotas))
		for _, us := range all {
			for _, u := range us {
				have[u.Class]++
			}
		}
		worst := 0.0
		for c, q := range quotas {
			if q > 0 && have[c] < q {
				short := float64(q-have[c]) / float64(q)
				ratio := float64(have[c]) / float64(q)
				need := short
				if ratio > 0 {
					need = short / ratio // batches so far produced `ratio` of the quota
				}
				worst = math.Max(worst, need)
			}
		}
		b.Log("%s: round %d, %d batches so far, units per class %v (quota %v)", b.DS.Name, round, next, have, quotas)
		if worst == 0 {
			return selectQuota(all, quotas), nil
		}
		batches = int(math.Ceil(worst*float64(next)*1.2)) + 1
		if batches > 400 {
			batches = 400
		}
	}
	return nil, fmt.Errorf("%s: quotas %v not met after 8 rounds", b.DS.Name, quotas)
}

func selectQuota(all [][]Unit, quotas []int) []Unit {
	left := append([]int(nil), quotas...)
	var out []Unit
	for _, us := range all {
		for _, u := range us {
			if left[u.Class] > 0 {
				left[u.Class]--
				out = append(out, u)
			}
		}
	}
	return out
}

// Quotas splits n into classes by shares (rounding so they sum to n).
func Quotas(n int, shares ...float64) []int {
	q := make([]int, len(shares))
	sum := 0
	for i, s := range shares {
		q[i] = int(math.Floor(float64(n) * s))
		sum += q[i]
	}
	for i := 0; sum < n; i = (i + 1) % len(q) {
		q[i]++
		sum++
	}
	return q
}

// Assign returns per-slot classes for one batch: drawn with probabilities proportional to shares.
func Assign(r *rand.Rand, k int, shares ...float64) []int {
	out := make([]int, k)
	tot := 0.0
	for _, s := range shares {
		tot += s
	}
	for i := range out {
		x := r.Float64() * tot
		for c, s := range shares {
			if x < s || c == len(shares)-1 {
				out[i] = c
				break
			}
			x -= s
		}
	}
	return out
}

// Pick returns a random element.
func Pick[T any](r *rand.Rand, xs []T) T { return xs[r.IntN(len(xs))] }

// PickN returns n distinct random elements.
func PickN[T any](r *rand.Rand, xs []T, n int) []T {
	idx := r.Perm(len(xs))
	out := make([]T, 0, n)
	for _, i := range idx[:min(n, len(xs))] {
		out = append(out, xs[i])
	}
	return out
}

// ---------------------------------------------------------------- item field helpers

// Str returns a non-empty string field, or "" when absent/wrong type.
func Str(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	s, _ := m[key].(string)
	return strings.TrimSpace(s)
}

// StrRaw returns a string field without trimming (outputs keep their whitespace); ok=false when absent.
func StrRaw(m map[string]any, key string) (string, bool) {
	if m == nil {
		return "", false
	}
	s, ok := m[key].(string)
	return s, ok
}

// IntField returns an integer field.
func IntField(m map[string]any, key string) (int, bool) {
	if m == nil {
		return 0, false
	}
	f, ok := m[key].(float64)
	if !ok || f != math.Trunc(f) {
		return 0, false
	}
	return int(f), true
}

// ByIndex orders items by their "i" field when every item has a distinct one in range; otherwise keeps order.
// Returns nil for an index that is missing.
func ByIndex(items []map[string]any, n int) []map[string]any {
	out := make([]map[string]any, n)
	ok := true
	for _, m := range items {
		i, has := IntField(m, "i")
		if !has || i < 0 || i >= n || out[i] != nil {
			ok = false
			break
		}
		out[i] = m
	}
	if ok {
		return out
	}
	out = make([]map[string]any, n)
	copy(out, items)
	return out
}

// ---------------------------------------------------------------- writing a dataset

// Stats is data/<ds>/synth_stats.json.
type Stats struct {
	Name         string         `json:"name"`
	N            int            `json:"n"`
	ClassCounts  []int          `json:"class_counts"`
	Options      []string       `json:"options"`
	Batches      map[string]int `json:"batches"`
	Drops        map[string]int `json:"drops"`
	CachedGens   int            `json:"cached_generations"`
	Seed         uint64         `json:"seed"`
	Time         string         `json:"time"`
	Programmatic bool           `json:"programmatic"`
	Model        string         `json:"model,omitempty"` // the generator model; empty for programmatic datasets
}

// Write builds a dataset and writes question.json, items.jsonl and synth_stats.json under dir.
func Write(ctx context.Context, d *Dataset, client *llm.Client, dir string, n int, seed uint64,
	logf func(string, ...any)) (*Stats, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	cache, err := llm.OpenCache(filepath.Join(dir, "synth_raw.jsonl"))
	if err != nil {
		return nil, err
	}
	defer cache.Close()
	b := &Builder{DS: d, LLM: client, Cache: cache, Seed: seed, Drops: map[string]int{}, Batch: map[string]int{}, Log: logf}
	units, err := d.Build(ctx, b, n)
	if err != nil {
		return nil, err
	}
	q := ds.QuestionFile{Name: d.Name, Category: d.Category, Type: d.Type, Options: d.Options, Pair: d.Pair, Notes: d.Notes}
	q.Question, err = ds.Marshal(d.Question)
	if err != nil {
		return nil, err
	}
	if err := q.Validate(); err != nil {
		return nil, err
	}
	items, err := Order(d, units, seed)
	if err != nil {
		return nil, err
	}
	counts := make([]int, len(d.Options))
	for _, it := range items {
		counts[it.Truth]++
	}
	if err := ds.WriteJSON(filepath.Join(dir, "question.json"), q); err != nil {
		return nil, err
	}
	if err := ds.WriteJSONL(filepath.Join(dir, "items.jsonl"), items); err != nil {
		return nil, err
	}
	st := &Stats{Name: d.Name, N: len(items), ClassCounts: counts, Options: d.Options, Batches: b.Batch, Drops: b.Drops,
		CachedGens: cache.Len(), Seed: seed, Time: time.Now().UTC().Format(time.RFC3339)}
	st.Programmatic = len(b.Batch) == 0
	if !st.Programmatic && client != nil {
		st.Model = client.Model
	}
	return st, ds.WriteJSON(filepath.Join(dir, "synth_stats.json"), st)
}

// Order flattens units into a shuffled stream (fixed seed) with ids, keeping a unit's members apart when asked,
// and refusing duplicate states (hunch keeps one example per state).
func Order(d *Dataset, units []Unit, seed uint64) ([]ds.Item, error) {
	type flat struct {
		m     Member
		group int
		state []byte
	}
	var fl []flat
	seen := map[string]bool{}
	for g, u := range units {
		for _, m := range u.Members {
			b, err := ds.Marshal(m.State)
			if err != nil {
				return nil, err
			}
			if seen[string(b)] {
				continue // a duplicate state: drop the later one
			}
			seen[string(b)] = true
			fl = append(fl, flat{m, g, b})
		}
	}
	h := fnv.New64a()
	h.Write([]byte("order/" + d.Name))
	r := rand.New(rand.NewPCG(seed, h.Sum64()))
	r.Shuffle(len(fl), func(i, j int) { fl[i], fl[j] = fl[j], fl[i] })
	if d.NoAdjacentGroups {
		for pass := 0; pass < 10; pass++ {
			fixed := true
			for i := 1; i < len(fl); i++ {
				if fl[i].group == fl[i-1].group {
					fixed = false
					j := r.IntN(len(fl))
					fl[i], fl[j] = fl[j], fl[i]
				}
			}
			if fixed {
				break
			}
		}
	}
	items := make([]ds.Item, len(fl))
	width := max(4, len(fmt.Sprint(len(fl))))
	for i, f := range fl {
		items[i] = ds.Item{ID: fmt.Sprintf("%s-%0*d", d.Name, width, i+1), State: json.RawMessage(f.state), Truth: f.m.Truth}
	}
	return items, nil
}

// ---------------------------------------------------------------- registry

// Registry lists every dataset, in the order of the benchmark plan.
func Registry() []*Dataset {
	return []*Dataset{ToolFailed(), OutputKind(), CustomerFrustration(), SecretNeedle(), PaymentFlip(), PoliterReply(),
		PassageAnswers(), CodeLookup(), RetryException(), CapitalCity(), IncidentSeverity()}
}

// Lookup returns datasets by name (all when names is empty).
func Lookup(names []string) ([]*Dataset, error) {
	all := Registry()
	if len(names) == 0 {
		return all, nil
	}
	by := map[string]*Dataset{}
	for _, d := range all {
		by[d.Name] = d
	}
	var out []*Dataset
	for _, n := range names {
		d, ok := by[n]
		if !ok {
			var known []string
			for k := range by {
				known = append(known, k)
			}
			sort.Strings(known)
			return nil, fmt.Errorf("unknown dataset %q (known: %s)", n, strings.Join(known, ", "))
		}
		out = append(out, d)
	}
	return out, nil
}

// noulQuestion builds a noul question object.
func noulQuestion(instructions, yes, no string) ds.OMap {
	return ds.OMap{kv("type", "noul"), kv("instructions", instructions), kv("criteria", ds.OMap{kv("true", yes), kv("false", no)})}
}

// choiceQuestion builds a choice question with criteria in option order.
func choiceQuestion(instructions string, opts []string, desc []string) ds.OMap {
	crit := ds.OMap{}
	for i, o := range opts {
		crit = append(crit, ds.KV{Key: o, Value: desc[i]})
	}
	return ds.OMap{kv("type", "choice"), kv("instructions", instructions), kv("criteria", crit)}
}

// scoreQuestion builds a score question; criteria is an ordered array (SPEC).
func scoreQuestion(instructions string, levels []string) ds.OMap {
	return ds.OMap{kv("type", "score"), kv("instructions", instructions), kv("criteria", levels)}
}

func kv(k string, v any) ds.KV { return ds.KV{Key: k, Value: v} }
