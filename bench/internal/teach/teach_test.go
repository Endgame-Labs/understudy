package teach

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/endgame-labs/understudy/bench/internal/ds"
)

func TestRequestAndParse(t *testing.T) {
	q := &ds.QuestionFile{Name: "lvl", Type: "score", Options: []string{"0", "1", "2"},
		Question: json.RawMessage(`{"type":"score","instructions":"x","criteria":["low","mid","high"]}`)}
	body, err := Request("m", q, json.RawMessage(`{"message":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(body), `{"model":"m","questions":{"lvl":{"type":"score"`) || !strings.Contains(string(body), `["low","mid","high"]`) {
		t.Fatalf("body %s", body)
	}
	resp := `{"model":"accounts/fireworks/routers/rune","answers":{"lvl":{"type":"score","score":1.9,"probabilities":{"0":0.05,"1":0.05,"2":0.9},"confidence":0.9}},"usage":{}}`
	_, p, model, err := ParseResponse(q, []byte(resp))
	if err != nil || ds.Argmax(p) != 2 || model != "accounts/fireworks/routers/rune" {
		t.Fatalf("%v %v %s", p, err, model)
	}
	if _, _, _, err := ParseResponse(q, []byte(`{"answers":{}}`)); err == nil {
		t.Fatal("accepted a response without the answer")
	}
}

// A dataset whose lead.<name>.jsonl is fully answered makes zero lead calls and leaves the file unchanged; a partial
// one asks only for the missing items and keeps the earlier answers.
func TestDatasetResumesFromLeadFile(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(`{"model":"m","answers":{"lvl":{"type":"score","score":0,"probabilities":{"0":0.9,"1":0.1}}}}`))
	}))
	defer srv.Close()
	dir := t.TempDir()
	q := ds.QuestionFile{Name: "lvl", Type: "score", Options: []string{"0", "1"},
		Question: json.RawMessage(`{"type":"score","instructions":"x","criteria":["low","high"]}`)}
	ds.WriteJSON(filepath.Join(dir, "question.json"), q)
	var items []ds.Item
	var old []ds.LeadRecord
	for _, id := range []string{"a", "b", "c"} {
		st := json.RawMessage(`{"message":"` + id + `"}`)
		items = append(items, ds.Item{ID: id, State: st})
		old = append(old, ds.LeadRecord{ID: id, Probs: []float64{0.2, 0.8}, Label: 1, Model: "old", StateSHA: ds.StateSHA(st)})
	}
	ds.WriteJSONL(filepath.Join(dir, "items.jsonl"), items)
	leadFile := filepath.Join(dir, "lead.jev.jsonl")
	ds.WriteJSONL(leadFile, old)
	before, _ := os.ReadFile(leadFile)

	l := New(Preset{Name: "jev", URL: srv.URL, Model: "m", Workers: 2})
	answered, failed, err := l.Dataset(context.Background(), dir)
	if err != nil || answered != 0 || failed != 0 || calls.Load() != 0 {
		t.Fatalf("answered=%d failed=%d calls=%d err=%v", answered, failed, calls.Load(), err)
	}
	after, err := os.ReadFile(leadFile)
	if err != nil || string(after) != string(before) {
		t.Fatalf("lead file changed or removed: %v", err)
	}

	// a partially answered file asks only for the missing items, and keeps the earlier answers
	ds.WriteJSONL(leadFile, old[:2])
	answered, _, err = l.Dataset(context.Background(), dir)
	if err != nil || answered != 1 || calls.Load() != 1 {
		t.Fatalf("answered=%d calls=%d err=%v", answered, calls.Load(), err)
	}
	got, _ := ds.ReadJSONL[ds.LeadRecord](filepath.Join(dir, "lead.jev.jsonl"))
	if len(got) != 3 || got[0].Model != "old" || got[1].Model != "old" || got[2].Model != "m" {
		t.Fatalf("%+v", got)
	}
}

func TestResolveWithKeepsKeyEnvUnderOverrides(t *testing.T) {
	dir := t.TempDir()
	p := dir + "/leads.toml"
	if err := os.WriteFile(p, []byte("[leads.sys]\nurl = \"https://a.example/v1/systemone\"\nmodel = \"m\"\nkey_env = \"SYS_KEY\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("UNDERSTUDY_LEADS", p)
	got, err := ResolveWith("sys", "https://b.example/v1/systemone", "m2")
	if err != nil || got.KeyEnv != "SYS_KEY" || got.URL != "https://b.example/v1/systemone" || got.Model != "m2" {
		t.Fatalf("%+v %v", got, err)
	}
	// a name the config lacks, with both overrides, needs no preset (and sends no key)
	got, err = ResolveWith("other", "https://c.example", "m3")
	if err != nil || got.KeyEnv != "" || got.URL != "https://c.example" {
		t.Fatalf("%+v %v", got, err)
	}
	if _, err := ResolveWith("other", "", ""); err == nil {
		t.Fatal("unknown lead without overrides accepted")
	}
}

// Fully explicit name, url and model never depend on the config: a missing $UNDERSTUDY_LEADS file or a malformed one
// does not block them.
func TestResolveWithExplicitNeedsNoConfig(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("UNDERSTUDY_LEADS", dir+"/nonexistent.toml")
	got, err := ResolveWith("unknown", "https://c.example/v1/systemone", "m3")
	if err != nil || got.KeyEnv != "" || got.URL != "https://c.example/v1/systemone" || got.Model != "m3" {
		t.Fatalf("%+v %v", got, err)
	}
	if _, err := ResolveWith("unknown", "", ""); err == nil {
		t.Fatal("a missing config with no overrides accepted")
	}
	bad := dir + "/bad.toml"
	if err := os.WriteFile(bad, []byte("[leads.x\nurl = \n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("UNDERSTUDY_LEADS", bad)
	if got, err = ResolveWith("unknown", "https://c.example", "m3"); err != nil || got.Model != "m3" {
		t.Fatalf("%+v %v", got, err)
	}
	if _, err := ResolveWith("unknown", "https://c.example", ""); err == nil {
		t.Fatal("a malformed config with one override accepted")
	}
}
