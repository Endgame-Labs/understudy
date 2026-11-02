package report

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/endgame-labs/understudy/bench/internal/ds"
	"github.com/endgame-labs/understudy/bench/internal/runner"
)

// A run made from data-private keeps no stream.jsonl; the report must work from stream.ids.jsonl and the private
// items/lead files, and must refuse to write a report next to a leaked state.
func TestReportFromTextFreeRun(t *testing.T) {
	root := t.TempDir()
	text := "The quick brown fox jumps over the lazy dog, twice, in the morning."
	qraw, _ := ds.Marshal(ds.OMap{{Key: "type", Value: "noul"}, {Key: "instructions", Value: "i"}, {Key: "criteria", Value: ds.OMap{{Key: "true", Value: "t"}, {Key: "false", Value: "f"}}}})
	q := ds.QuestionFile{Name: "d", Category: "real_test", Type: "noul", Options: []string{"false", "true"}, Question: qraw}
	data := filepath.Join(root, "data-private", "d")
	run := filepath.Join(root, "runs", "r")
	var items []ds.Item
	var recs []ds.LeadRecord
	type idRow struct {
		ID        string    `json:"id"`
		LeadProbs []float64 `json:"lead_probs"`
	}
	var ids []idRow
	var decs []ds.Decision
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("d-%04d", i+1)
		st := []byte(fmt.Sprintf(`{"text":"%s %d"}`, text, i))
		items = append(items, ds.Item{ID: id, State: st, Truth: i % 2})
		recs = append(recs, ds.LeadRecord{ID: id, Probs: []float64{0.3, 0.7}, Label: 1, StateSHA: ds.StateSHA(st)})
		ids = append(ids, idRow{id, []float64{0.3, 0.7}})
		decs = append(decs, ds.Decision{ID: id, AnsweredBy: "lead", Label: 1, Probs: []float64{0.3, 0.7}})
	}
	ds.WriteJSONL(filepath.Join(data, "items.jsonl"), items)
	ds.WriteJSONL(filepath.Join(data, "lead.jev.jsonl"), recs)
	ds.WriteJSON(filepath.Join(run, "d", "question.json"), q)
	ds.WriteJSONL(filepath.Join(run, "d", runner.StreamIDsFile), ids)
	ds.WriteJSONL(filepath.Join(run, "d", "decisions.hunch.jsonl"), decs)
	ds.WriteJSON(filepath.Join(run, "run.json"), runner.RunMeta{Run: "r", Lead: "jev", Datasets: []string{"d"}, Systems: []string{"hunch"},
		Modes: []string{"replay"}, DataDir: "data-private", NoText: true})
	r, err := Write(root, "r", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Datasets) != 1 || r.Datasets[0].N != 6 || r.Datasets[0].Systems[0].Metrics == nil {
		t.Fatalf("%+v", r.Datasets)
	}
	for _, f := range []string{"results.json", "REPORT.md"} {
		b, err := os.ReadFile(filepath.Join(run, f))
		if err != nil || strings.Contains(string(b), "quick brown") {
			t.Fatalf("%s: %v", f, err)
		}
	}
	// a state-bearing stream.jsonl left behind makes the report fail loudly
	ds.WriteJSONL(filepath.Join(run, "d", "stream.jsonl"), []ds.StreamItem{{ID: "d-0001", State: items[0].State, LeadProbs: []float64{0.3, 0.7}}})
	if _, err := Write(root, "r", ""); err == nil {
		t.Fatal("report accepted a leftover stream.jsonl in a text-free run")
	}
}
