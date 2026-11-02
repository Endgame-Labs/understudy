// Package teach asks a systemone lead (named in the lead config file) each dataset item's question and records the answers
// as data/<ds>/lead.<lead>.jsonl (SPEC.md), resumably by id.
package teach

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dorkitude/understudy/bench/internal/ds"
	"github.com/dorkitude/understudy/bench/internal/leadcfg"
	"github.com/dorkitude/understudy/bench/internal/netx"
)

// Preset is a lead's endpoint. KeyEnv names the environment variable holding a bearer key (empty: send none).
type Preset struct {
	Name, URL, Model, KeyEnv string
	Workers                  int
}

// ResolveWith reads the named preset from the lead config file ($UNDERSTUDY_LEADS or ~/.config/understudy/leads.toml,
// the same file understudy's --lead reads) and applies the url and model overrides on top of it, so a preset's key_env
// survives them. No lead is built in. With both overrides, a missing, unreadable or malformed config file, or a name it
// does not define, resolves anyway (no key).
func ResolveWith(name, url, model string) (Preset, error) {
	explicit := Preset{Name: name, URL: url, Model: model, Workers: 4}
	both := url != "" && model != ""
	c, err := leadcfg.Load()
	if err != nil {
		if both { // an unreadable config must not block a fully explicit lead
			return explicit, nil
		}
		return Preset{}, err
	}
	l, err := c.Lead(name)
	if err != nil {
		if errors.Is(err, leadcfg.ErrUnknownLead) && both {
			return explicit, nil
		}
		return Preset{}, err
	}
	if l.Kind != "systemone" {
		return Preset{}, fmt.Errorf("lead %q is kind %q: the bench teaches System One leads; label with a chat lead through `understudy judge --lead %s` instead", name, l.Kind, name)
	}
	if url != "" {
		l.URL = url
	}
	if model != "" {
		l.Model = model
	}
	if l.URL == "" || l.Model == "" {
		return Preset{}, fmt.Errorf("lead %q needs a url and a model in the lead config file (or --url and --model)", name)
	}
	return Preset{Name: name, URL: l.URL, Model: l.Model, KeyEnv: l.KeyEnv, Workers: 4}, nil
}

// Stats counts calls.
type Stats struct {
	Done, Skipped, Failed, Retries atomic.Int64
}

// Lead sends systemone requests.
type Lead struct {
	Preset
	HTTP  *netx.Client
	Stats Stats
	Log   func(string, ...any)
}

// New returns a lead with retries (408/429/5xx; Retry-After honored).
func New(p Preset) *Lead {
	h := netx.New(3*time.Minute, 8)
	h.Backoff = 2 * time.Second
	if b := leadcfg.Bearer(p.KeyEnv); b != "" {
		h.Headers = map[string]string{"Authorization": b}
	}
	t := &Lead{Preset: p, HTTP: h, Log: func(string, ...any) {}}
	h.OnRetry = func(attempt, status int, err error, sleep time.Duration) {
		t.Stats.Retries.Add(1)
	}
	return t
}

// Request builds the systemone body for one item: one question, named after the dataset.
func Request(model string, q *ds.QuestionFile, state json.RawMessage) ([]byte, error) {
	return ds.Marshal(ds.OMap{{Key: "model", Value: model},
		{Key: "questions", Value: ds.OMap{{Key: q.Name, Value: q.Question}}}, {Key: "state", Value: state}})
}

// ParseResponse extracts the dataset question's answer and its probabilities in class order.
func ParseResponse(q *ds.QuestionFile, body []byte) (json.RawMessage, []float64, string, error) {
	var resp struct {
		Model   string                     `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, nil, "", fmt.Errorf("response: %w", err)
	}
	a, ok := resp.Answers[q.Name]
	if !ok {
		return nil, nil, "", fmt.Errorf("response has no answer for %q", q.Name)
	}
	p, err := ds.ProbsFromAnswer(q.Type, q.Options, a)
	return a, p, resp.Model, err
}

// Dataset teaches every item of data/<name> not yet answered. Returns (answered now, failed).
func (t *Lead) Dataset(ctx context.Context, dir string) (int, int, error) {
	q, err := ds.ReadQuestion(filepath.Join(dir, "question.json"))
	if err != nil {
		return 0, 0, err
	}
	items, err := ds.ReadJSONL[ds.Item](filepath.Join(dir, "items.jsonl"))
	if err != nil {
		return 0, 0, err
	}
	path := filepath.Join(dir, "lead."+t.Name+".jsonl")
	// Resume from lead.<name>.jsonl so already paid-for answers are never asked again.
	have, err := ds.ReadJSONL[ds.LeadRecord](path)
	if err != nil {
		return 0, 0, err
	}
	sha := map[string]string{}
	for _, it := range items {
		sha[it.ID] = ds.StateSHA(it.State)
	}
	done := map[string]bool{}
	stale := 0
	for _, r := range have {
		if r.StateSHA == sha[r.ID] && r.StateSHA != "" {
			done[r.ID] = true
		} else {
			stale++
		}
	}
	if stale > 0 {
		t.Log("%s: ignoring %d %s answers whose state no longer matches items.jsonl", q.Name, stale, t.Name)
	}
	app, err := ds.OpenAppender(path)
	if err != nil {
		return 0, 0, err
	}
	var todo []ds.Item
	for _, it := range items {
		if !done[it.ID] {
			todo = append(todo, it)
		} else {
			t.Stats.Skipped.Add(1)
		}
	}
	var answered, failed atomic.Int64
	ch := make(chan ds.Item)
	var wg sync.WaitGroup
	for w := 0; w < max(1, t.Workers); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for it := range ch {
				body, err := Request(t.Model, q, it.State)
				if err != nil {
					t.Log("%s: %v", it.ID, err)
					failed.Add(1)
					continue
				}
				t0 := time.Now()
				resp, _, err := t.HTTP.PostJSON(ctx, t.URL, body)
				lat := time.Since(t0).Milliseconds()
				var rec ds.LeadRecord
				if err == nil {
					var model string
					rec.Answer, rec.Probs, model, err = ParseResponse(q, resp)
					rec.Model = model
					if rec.Model == "" {
						rec.Model = t.Model
					}
				}
				if err != nil {
					t.Log("%s %s: %v", t.Name, it.ID, err)
					failed.Add(1)
					t.Stats.Failed.Add(1)
					continue
				}
				rec.ID, rec.Label, rec.LatencyMS, rec.StateSHA = it.ID, ds.Argmax(rec.Probs), lat, sha[it.ID]
				if err := app.Append(rec); err != nil {
					t.Log("append: %v", err)
					failed.Add(1)
					continue
				}
				answered.Add(1)
				t.Stats.Done.Add(1)
			}
		}()
	}
	for _, it := range todo {
		select {
		case ch <- it:
		case <-ctx.Done():
		}
		if ctx.Err() != nil {
			break
		}
	}
	close(ch)
	wg.Wait()
	app.Close()
	if ctx.Err() != nil {
		return int(answered.Load()), int(failed.Load()), ctx.Err()
	}
	// rewrite in stream order (one record per id) so the file reads like items.jsonl
	all, err := ds.ReadJSONL[ds.LeadRecord](path)
	if err != nil {
		return 0, 0, err
	}
	pos := map[string]int{}
	for i, it := range items {
		pos[it.ID] = i
	}
	byID := map[string]ds.LeadRecord{}
	for _, r := range all {
		if _, ok := pos[r.ID]; ok && r.StateSHA == sha[r.ID] {
			byID[r.ID] = r
		}
	}
	rows := make([]ds.LeadRecord, 0, len(byID))
	for _, r := range byID {
		rows = append(rows, r)
	}
	sort.Slice(rows, func(i, j int) bool { return pos[rows[i].ID] < pos[rows[j].ID] })
	return int(answered.Load()), int(failed.Load()), ds.WriteJSONL(path, rows)
}
