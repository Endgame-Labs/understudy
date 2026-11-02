package ds

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProbsFromAnswer(t *testing.T) {
	p, err := ProbsFromAnswer(Noul, []string{"false", "true"}, json.RawMessage(`{"type":"noul","noul":0.8}`))
	if err != nil || len(p) != 2 || p[0] < 0.199 || p[0] > 0.201 || p[1] != 0.8 {
		t.Fatalf("noul: %v %v", p, err)
	}
	p, err = ProbsFromAnswer(Choice, []string{"b", "a"}, json.RawMessage(`{"type":"choice","choice":"a","probabilities":{"a":0.7,"b":0.3}}`))
	if err != nil || p[0] != 0.3 || p[1] != 0.7 {
		t.Fatalf("choice order: %v %v", p, err)
	}
	p, err = ProbsFromAnswer(Score, []string{"0", "1", "2"}, json.RawMessage(`{"type":"score","probabilities":{"0":0.1,"1":0.2,"2":0.7}}`))
	if err != nil || Argmax(p) != 2 {
		t.Fatalf("score: %v %v", p, err)
	}
	bad := []string{`{"type":"noul"}`, `{"noul":1.5}`, `{"type":"choice","noul":0.3}`}
	for _, b := range bad {
		if _, err := ProbsFromAnswer(Noul, []string{"false", "true"}, json.RawMessage(b)); err == nil {
			t.Errorf("accepted %s", b)
		}
	}
	if _, err := ProbsFromAnswer(Choice, []string{"a", "b"}, json.RawMessage(`{"probabilities":{"a":0.2,"c":0.8}}`)); err == nil {
		t.Error("accepted a missing option")
	}
	if _, err := ProbsFromAnswer(Score, []string{"0", "1"}, json.RawMessage(`{"probabilities":{"0":0.2,"1":0.2}}`)); err == nil {
		t.Error("accepted probabilities summing to 0.4")
	}
}

func TestOMapOrderAndValidate(t *testing.T) {
	q := OMap{{"type", "choice"}, {"instructions", "x <y>"}, {"criteria", OMap{{"z", "last?"}, {"a", "first?"}}}}
	b, err := Marshal(q)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `{"z":"last?","a":"first?"}`) || !strings.Contains(string(b), "<y>") {
		t.Fatalf("order or escaping lost: %s", b)
	}
	f := QuestionFile{Name: "n", Type: "choice", Options: []string{"z", "a"}, Question: b}
	if err := f.Validate(); err != nil {
		t.Fatal(err)
	}
	f.Options = []string{"z", "q"}
	if f.Validate() == nil {
		t.Fatal("accepted an option without a criterion")
	}
	s := QuestionFile{Name: "s", Type: "score", Options: []string{"0", "1"},
		Question: json.RawMessage(`{"type":"score","criteria":{"0":"a","1":"b"}}`)}
	if s.Validate() == nil {
		t.Fatal("accepted score criteria as an object")
	}
}

func TestReadJSONLTruncatedTail(t *testing.T) {
	p := filepath.Join(t.TempDir(), "x.jsonl")
	os.WriteFile(p, []byte("{\"id\":\"a\",\"truth\":1}\n\n{\"id\":\"b\",\"tru"), 0o644)
	rows, err := ReadJSONL[Item](p)
	if err != nil || len(rows) != 1 || rows[0].ID != "a" {
		t.Fatalf("%v %v", rows, err)
	}
	a, err := OpenAppender(p)
	if err != nil {
		t.Fatal(err)
	}
	a.Append(Item{ID: "c", Truth: 0})
	a.Close()
	rows, _ = ReadJSONL[Item](p)
	if len(rows) != 2 || rows[1].ID != "c" {
		t.Fatalf("append after a partial line: %v", rows)
	}
	os.WriteFile(p, []byte("{bad}\n{\"id\":\"a\"}\n"), 0o644)
	if _, err := ReadJSONL[Item](p); err == nil {
		t.Fatal("accepted a bad middle line")
	}
}
