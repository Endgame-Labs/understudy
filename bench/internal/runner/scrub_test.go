package runner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dorkitude/understudy/bench/internal/ds"
)

const textA = "The quick brown fox jumps over the lazy dog, twice, in the morning."

func fixture(t *testing.T) (outDir, items string) {
	t.Helper()
	dir := t.TempDir()
	outDir = filepath.Join(dir, "runs", "r", "d")
	os.MkdirAll(filepath.Join(outDir, "snapshot.understudent-deep", "fits"), 0o755)
	os.WriteFile(filepath.Join(outDir, "snapshot.understudent-deep", "snapshot.json"), []byte(textA), 0o644)
	items = filepath.Join(dir, "data-private", "d", "items.jsonl")
	if err := ds.WriteJSONL(items, []ds.Item{{ID: "d-0001", State: []byte(`{"text":"` + textA + `"}`), Truth: 1}}); err != nil {
		t.Fatal(err)
	}
	st := []ds.StreamItem{{ID: "d-0001", State: []byte(`{"text":"` + textA + `"}`), LeadProbs: []float64{0.2, 0.8}}}
	ds.WriteJSONL(filepath.Join(outDir, "stream.jsonl"), st)
	ds.WriteJSONL(filepath.Join(outDir, "train.jsonl"), []TrainLine{{"d-0001", map[string]string{"text": textA}, []float64{0.2, 0.8}}})
	ds.WriteJSONL(filepath.Join(outDir, "operate.jsonl"), []OperateLine{{"d-0001", map[string]string{"text": textA}}})
	us := 12.0
	mode := "symmetric"
	ds.WriteJSONL(filepath.Join(outDir, "decisions.hunch.jsonl"), []ds.Decision{{ID: "d-0001", AnsweredBy: "lead", Label: 1, Probs: []float64{0.2, 0.8}, Mode: &mode, UnderstudentUS: &us}})
	os.WriteFile(filepath.Join(outDir, "timing.hunch.json"), []byte(`{"system":"hunch","wall_s":1}`), 0o644)
	return
}

func TestScrubRemovesStatesAndKeepsDecisions(t *testing.T) {
	out, items := fixture(t)
	if err := Scrub(out, items); err != nil {
		t.Fatal(err)
	}
	for _, gone := range []string{"stream.jsonl", "train.jsonl", "operate.jsonl", "snapshot.understudent-deep"} {
		if _, err := os.Stat(filepath.Join(out, gone)); err == nil {
			t.Errorf("%s was not deleted", gone)
		}
	}
	ids, err := os.ReadFile(filepath.Join(out, StreamIDsFile))
	if err != nil || strings.Contains(string(ids), "quick") || !strings.Contains(string(ids), "lead_probs") {
		t.Fatalf("ids file: %s %v", ids, err)
	}
	if _, err := os.Stat(filepath.Join(out, "decisions.hunch.jsonl")); err != nil {
		t.Fatal("decisions removed")
	}
	// scrubbing again (a rerun's deferred scrub) is fine
	if err := Scrub(out, items); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyNoTextCatchesLeaks(t *testing.T) {
	out, items := fixture(t)
	if err := Scrub(out, items); err != nil {
		t.Fatal(err)
	}
	// a stderr log echoing a state
	log := filepath.Join(out, "hunch.stderr.log")
	os.WriteFile(log, []byte("error while reading: "+textA+"\n"), 0o644)
	if err := VerifyNoText(out, items); err == nil || !strings.Contains(err.Error(), "state text") {
		t.Fatalf("leak not caught: %v", err)
	}
	os.Remove(log)
	// a decision with an extra text field
	os.WriteFile(filepath.Join(out, "decisions.x.jsonl"), []byte(`{"id":"d-0001","answered_by":"lead","label":1,"probs":[0.2,0.8],"state":"`+textA+`"}`+"\n"), 0o644)
	if err := VerifyNoText(out, items); err == nil || !strings.Contains(err.Error(), "plain decision") {
		t.Fatalf("extra field not caught: %v", err)
	}
	os.WriteFile(filepath.Join(out, "decisions.x.jsonl"), []byte(`{"id":"`+textA+`","answered_by":"lead","label":1,"probs":[0.2,0.8]}`+"\n"), 0o644)
	if err := VerifyNoText(out, items); err == nil || !strings.Contains(err.Error(), "suspicious id") {
		t.Fatalf("text id not caught: %v", err)
	}
	os.Remove(filepath.Join(out, "decisions.x.jsonl"))
	if err := VerifyNoText(out, items); err != nil {
		t.Fatal(err)
	}
	// a leftover stream is an error
	os.WriteFile(filepath.Join(out, "stream.jsonl"), []byte("{}\n"), 0o644)
	if err := VerifyNoText(out, items); err == nil {
		t.Fatal("stream.jsonl accepted")
	}
}

func TestDataPathAndPrivate(t *testing.T) {
	if DataPath("/r", "", "x") != "/r/data/x" || DataPath("/r", "data-private", "x") != "/r/data-private/x" || DataPath("/r", "/abs/data-private", "x") != "/abs/data-private/x" {
		t.Fatal("DataPath")
	}
	if !IsPrivateDataDir("data-private") || !IsPrivateDataDir("/a/b/data-private/") || IsPrivateDataDir("data") {
		t.Fatal("IsPrivateDataDir")
	}
}

func TestVerifyNoTextAcceptsPerResponseIDsAndHunchField(t *testing.T) {
	out, items := fixture(t)
	if err := Scrub(out, items); err != nil {
		t.Fatal(err)
	}
	good := `{"id":"judge_rb2_safety_per-0001#3","answered_by":"lead","label":1,"probs":[0.2,0.8],"hunch_answered_by":"backend"}` + "\n"
	os.WriteFile(filepath.Join(out, "decisions.hunch.jsonl"), []byte(good), 0o644)
	if err := VerifyNoText(out, items); err != nil {
		t.Fatalf("per-response id / hunch_answered_by rejected: %v", err)
	}
	// still strict: unknown fields and text-like ids fail
	bad := `{"id":"d-0001","answered_by":"lead","label":1,"probs":[0.2,0.8],"hunch_answered_by":"backend","extra":"x"}` + "\n"
	os.WriteFile(filepath.Join(out, "decisions.hunch.jsonl"), []byte(bad), 0o644)
	if err := VerifyNoText(out, items); err == nil || !strings.Contains(err.Error(), "plain decision") {
		t.Fatalf("unknown field accepted: %v", err)
	}
	bad = `{"id":"has space #1","answered_by":"lead","label":1,"probs":[0.2,0.8]}` + "\n"
	os.WriteFile(filepath.Join(out, "decisions.hunch.jsonl"), []byte(bad), 0o644)
	if err := VerifyNoText(out, items); err == nil || !strings.Contains(err.Error(), "suspicious id") {
		t.Fatalf("spaced id accepted: %v", err)
	}
}

const checklistSummary = `{
  "final": {
    "report": {
      "checklist": {
        "rules": [
          {"feature": "\"quic\"", "class": "class 1", "confidence": 0.9, "rows": 5, "hits": 5}
        ],
        "default": "class 0",
        "listing": "1. IF \"quic\" THEN class 1   [conf 0.900, 5/5 rows]\ndefault (abstain, never answers): class 0\n"
      },
      "readout": "checklist"
    }
  }
}
`

func TestScrubStripsChecklistRules(t *testing.T) {
	out, items := fixture(t)
	sum := filepath.Join(out, "summary.understudent-checklist.json")
	os.WriteFile(sum, []byte(checklistSummary), 0o644)
	// the stderr log echoes the summary pretty-printed, plus a stray listing line
	logText := "$ understudy replay --understudent checklist\n" +
		"{\n  \"final\": {\n    \"report\": {\n      \"checklist\": {\n        \"rules\": [\n          {\n            \"feature\": \"\\\"quic\\\"\",\n            \"rows\": 5\n          }\n        ],\n" +
		"        \"default\": \"class 0\",\n        \"listing\": \"1. IF \\\"quic\\\" THEN class 1\"\n      },\n      \"readout\": \"checklist\"\n    }\n  }\n}\n2. IF word:fox THEN class 0\n"
	log := filepath.Join(out, "understudent-checklist.stderr.log")
	os.WriteFile(log, []byte(logText), 0o644)
	if err := Scrub(out, items); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{sum, log} {
		b, _ := os.ReadFile(p)
		for _, bad := range []string{"quic", "IF ", `"rules"`, `"listing"`, "fox"} {
			if strings.Contains(string(b), bad) {
				t.Errorf("%s still contains %q:\n%s", filepath.Base(p), bad, b)
			}
		}
		if !strings.Contains(string(b), `"readout"`) {
			t.Errorf("%s lost unrelated content", filepath.Base(p))
		}
	}
	// results.json is handled by StripChecklistsFile
	res := filepath.Join(t.TempDir(), "results.json")
	os.WriteFile(res, []byte(checklistSummary), 0o644)
	if err := StripChecklistsFile(res); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(res); strings.Contains(string(b), "quic") || !strings.Contains(string(b), `"default"`) {
		t.Fatalf("results.json: %s", b)
	}
	// a leftover listing is a verification error
	os.WriteFile(log, []byte("1. IF word:cat THEN class 1\n"), 0o644)
	if err := VerifyNoText(out, items); err == nil || !strings.Contains(err.Error(), "checklist rules") {
		t.Fatalf("leftover listing accepted: %v", err)
	}
}
