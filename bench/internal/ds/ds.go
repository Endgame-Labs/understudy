// Package ds holds the file formats shared by understudy-bench and the replay runners (SPEC.md, "Files"): question.json,
// items.jsonl, lead.<lead>.jsonl, stream.jsonl and decisions.<system>.jsonl, plus JSONL helpers.
package ds

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// Wire types.
const (
	Noul   = "noul"
	Choice = "choice"
	Score  = "score"
)

// KV is one member of an ordered JSON object.
type KV struct {
	Key   string
	Value any
}

// OMap is a JSON object whose members are written in order (Go maps are not).
type OMap []KV

func (m OMap) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, kv := range m {
		if i > 0 {
			b.WriteByte(',')
		}
		k, err := json.Marshal(kv.Key)
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		v, err := marshalNoEscape(kv.Value)
		if err != nil {
			return nil, err
		}
		b.Write(v)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

func marshalNoEscape(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(b.Bytes(), "\n"), nil
}

// Marshal is json.Marshal without HTML escaping (states contain <, > and & often).
func Marshal(v any) ([]byte, error) { return marshalNoEscape(v) }

// QuestionFile is data/<ds>/question.json. Question is kept raw so its criteria keep the order they were written in.
type QuestionFile struct {
	Name     string          `json:"name"`
	Category string          `json:"category"`
	Type     string          `json:"type"`
	Options  []string        `json:"options"`
	Question json.RawMessage `json:"question"`
	Pair     []string        `json:"pair"`
	Notes    string          `json:"notes"`
}

// K is the number of classes.
func (q *QuestionFile) K() int { return len(q.Options) }

// Validate checks the question file against SPEC.md.
func (q *QuestionFile) Validate() error {
	if q.Name == "" {
		return errors.New("question.json: no name")
	}
	var inner struct {
		Type     string          `json:"type"`
		Criteria json.RawMessage `json:"criteria"`
	}
	if err := json.Unmarshal(q.Question, &inner); err != nil {
		return fmt.Errorf("question.json %s: question: %w", q.Name, err)
	}
	if inner.Type != q.Type {
		return fmt.Errorf("question.json %s: type %q but question.type %q", q.Name, q.Type, inner.Type)
	}
	switch q.Type {
	case Noul:
		if len(q.Options) != 2 || q.Options[0] != "false" || q.Options[1] != "true" {
			return fmt.Errorf("question.json %s: noul options must be [false true], got %v", q.Name, q.Options)
		}
	case Choice:
		var crit map[string]any
		if err := json.Unmarshal(inner.Criteria, &crit); err != nil {
			return fmt.Errorf("question.json %s: choice criteria must be an object: %w", q.Name, err)
		}
		if len(crit) != len(q.Options) {
			return fmt.Errorf("question.json %s: %d criteria for %d options", q.Name, len(crit), len(q.Options))
		}
		for _, o := range q.Options {
			if _, ok := crit[o]; !ok {
				return fmt.Errorf("question.json %s: option %q has no criterion", q.Name, o)
			}
		}
	case Score:
		var crit []string
		if err := json.Unmarshal(inner.Criteria, &crit); err != nil {
			return fmt.Errorf("question.json %s: score criteria must be an ordered array: %w", q.Name, err)
		}
		if len(crit) != len(q.Options) {
			return fmt.Errorf("question.json %s: %d levels for %d options", q.Name, len(crit), len(q.Options))
		}
	default:
		return fmt.Errorf("question.json %s: unknown type %q", q.Name, q.Type)
	}
	if q.Pair != nil && len(q.Pair) != 2 {
		return fmt.Errorf("question.json %s: pair must be null or two field names", q.Name)
	}
	return nil
}

// Item is one line of items.jsonl. NoTruth marks an item with no ground truth (`truth` absent from the file: lead-only
// datasets); Truth is then 0 and meaningless.
type Item struct {
	ID      string          `json:"id"`
	State   json.RawMessage `json:"state"`
	Truth   int             `json:"truth"`
	NoTruth bool            `json:"-"`
}

type itemWire struct {
	ID    string          `json:"id"`
	State json.RawMessage `json:"state"`
	Truth *int            `json:"truth,omitempty"`
}

// MarshalJSON omits truth when NoTruth is set.
func (it Item) MarshalJSON() ([]byte, error) {
	w := itemWire{ID: it.ID, State: it.State}
	if !it.NoTruth {
		w.Truth = &it.Truth
	}
	return Marshal(w)
}

// UnmarshalJSON accepts a missing truth (NoTruth).
func (it *Item) UnmarshalJSON(b []byte) error {
	var w itemWire
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	*it = Item{ID: w.ID, State: w.State, NoTruth: w.Truth == nil}
	if w.Truth != nil {
		it.Truth = *w.Truth
	}
	return nil
}

// LeadRecord is one line of lead.<lead>.jsonl.
type LeadRecord struct {
	ID        string          `json:"id"`
	Probs     []float64       `json:"probs"`
	Label     int             `json:"label"`
	Answer    json.RawMessage `json:"answer"`
	LatencyMS int64           `json:"latency_ms"`
	Model     string          `json:"model"`
	// StateSHA (extra to SPEC) ties the answer to the state it answered, so a regenerated items.jsonl never reuses
	// another state's answer under the same id.
	StateSHA string `json:"state_sha,omitempty"`
}

// StateSHA hashes a state's JSON.
func StateSHA(state json.RawMessage) string {
	h := sha256.Sum256(state)
	return hex.EncodeToString(h[:8])
}

// LeadPath is the path of a lead's answers for a dataset: lead.<name>.jsonl.
func LeadPath(dir, name string) string {
	return filepath.Join(dir, "lead."+name+".jsonl")
}

// ReadDecisions reads a decisions.<system>.jsonl.
func ReadDecisions(path string) ([]Decision, error) {
	return ReadJSONL[Decision](path)
}

// StreamItem is one line of runs/<run>/<ds>/stream.jsonl (no truth).
type StreamItem struct {
	ID        string          `json:"id"`
	State     json.RawMessage `json:"state"`
	LeadProbs []float64       `json:"lead_probs"`
}

// Decision is one line of decisions.<system>.jsonl.
type Decision struct {
	ID                string    `json:"id"`
	AnsweredBy        string    `json:"answered_by"`
	Label             int       `json:"label"`
	Probs             []float64 `json:"probs"`
	UnderstudentLabel *int      `json:"understudent_label"`
	UnderstudentProbs []float64 `json:"understudent_probs"`
	Eligible          bool      `json:"eligible"`
	Novel             bool      `json:"novel"`
	Demoted           bool      `json:"demoted"`
	Mode              *string   `json:"mode"`
	Audited           bool      `json:"audited"`
	Readout           *string   `json:"readout"`
	UnderstudentUS    *float64  `json:"understudent_us"`
	// HunchAnsweredBy is written by the hunch runner only (who answered inside hunch's proxy: backend/proxy).
	HunchAnsweredBy *string `json:"hunch_answered_by,omitempty"`
}

// Argmax returns the index of the largest value (the first on ties).
func Argmax(p []float64) int {
	best := 0
	for i, x := range p {
		if x > p[best] {
			best = i
		}
	}
	return best
}

// ProbsFromAnswer turns a systemone answer object into class probabilities in class-index order (SPEC "Wire
// types"): noul {"noul": p_yes} -> [1-p, p]; choice {"probabilities": {opt: p}} in options order; score
// {"probabilities": {"0": p, ...}}. Probabilities must be finite, in [0, 1] and sum to about 1; they are
// renormalized.
func ProbsFromAnswer(qtype string, options []string, answer json.RawMessage) ([]float64, error) {
	var a struct {
		Type  string             `json:"type"`
		Noul  *float64           `json:"noul"`
		Probs map[string]float64 `json:"probabilities"`
	}
	if err := json.Unmarshal(answer, &a); err != nil {
		return nil, fmt.Errorf("answer: %w", err)
	}
	if a.Type != "" && a.Type != qtype {
		return nil, fmt.Errorf("answer type %q for a %s question", a.Type, qtype)
	}
	var p []float64
	switch qtype {
	case Noul:
		if a.Noul == nil {
			return nil, errors.New("noul answer without a noul probability")
		}
		y := *a.Noul
		if !(y >= 0 && y <= 1) {
			return nil, fmt.Errorf("noul probability %v out of [0, 1]", y)
		}
		return []float64{1 - y, y}, nil
	case Choice:
		if len(a.Probs) != len(options) {
			return nil, fmt.Errorf("choice answer has %d probabilities for %d options", len(a.Probs), len(options))
		}
		for _, o := range options {
			x, ok := a.Probs[o]
			if !ok {
				return nil, fmt.Errorf("choice answer has no probability for %q", o)
			}
			p = append(p, x)
		}
	case Score:
		if len(a.Probs) != len(options) {
			return nil, fmt.Errorf("score answer has %d probabilities for %d levels", len(a.Probs), len(options))
		}
		for i := range options {
			x, ok := a.Probs[strconv.Itoa(i)]
			if !ok {
				return nil, fmt.Errorf("score answer has no probability for level %d", i)
			}
			p = append(p, x)
		}
	default:
		return nil, fmt.Errorf("unknown type %q", qtype)
	}
	s := 0.0
	for _, x := range p {
		if math.IsNaN(x) || x < 0 || x > 1 {
			return nil, fmt.Errorf("probability %v out of [0, 1]", x)
		}
		s += x
	}
	if math.Abs(s-1) > 0.02 {
		return nil, fmt.Errorf("probabilities sum to %v", s)
	}
	for i := range p {
		p[i] /= s
	}
	return p, nil
}

// ---------------------------------------------------------------- files and JSONL

// FindRoot walks up from dir to the understudy repository root (the directory holding SPEC.md).
func FindRoot(dir string) (string, error) {
	d, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(d, "SPEC.md")); err == nil {
			return d, nil
		}
		parent := filepath.Dir(d)
		if parent == d {
			return "", fmt.Errorf("no SPEC.md above %s: pass --root", dir)
		}
		d = parent
	}
}

// ReadJSON reads one JSON file.
func ReadJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// WriteJSON writes v indented, atomically.
func WriteJSON(path string, v any) error {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return err
	}
	return WriteFileAtomic(path, b.Bytes())
}

// WriteFileAtomic writes via a temporary file and a rename.
func WriteFileAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ReadQuestion reads and validates a question.json.
func ReadQuestion(path string) (*QuestionFile, error) {
	var q QuestionFile
	if err := ReadJSON(path, &q); err != nil {
		return nil, err
	}
	return &q, q.Validate()
}

// ReadJSONL decodes every non-empty line of a JSONL file. A missing file gives no rows. A truncated last line (an
// interrupted append) is skipped; any other bad line is an error.
func ReadJSONL[T any](path string) ([]T, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []T
	r := bufio.NewReaderSize(f, 1<<20)
	line := 0
	for {
		b, err := r.ReadBytes('\n')
		if len(bytes.TrimSpace(b)) > 0 {
			line++
			var v T
			if jerr := json.Unmarshal(b, &v); jerr != nil {
				if err == io.EOF { // last line without a newline: an interrupted write
					break
				}
				return nil, fmt.Errorf("%s line %d: %w", path, line, jerr)
			}
			out = append(out, v)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// WriteJSONL writes rows, one JSON object per line, atomically.
func WriteJSONL[T any](path string, rows []T) error {
	var b bytes.Buffer
	for _, r := range rows {
		line, err := Marshal(r)
		if err != nil {
			return err
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	return WriteFileAtomic(path, b.Bytes())
}

// Appender appends JSON lines to a file, safe for concurrent use.
type Appender struct {
	f  *os.File
	ch chan []byte
	mu chan struct{}
}

// OpenAppender opens path for appending. A partial last line (an interrupted write) is truncated away.
func OpenAppender(path string) (*Appender, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	if st, err := f.Stat(); err == nil && st.Size() > 0 {
		// drop a partial last line (an interrupted write) so the next record starts cleanly
		size := st.Size()
		buf := make([]byte, 64<<10)
		end := size
		for end > 0 {
			start := max(0, end-int64(len(buf)))
			chunk := buf[:end-start]
			if _, err := f.ReadAt(chunk, start); err != nil {
				break
			}
			if i := bytes.LastIndexByte(chunk, '\n'); i >= 0 {
				end = start + int64(i) + 1
				break
			}
			end = start
		}
		if end < size {
			if err := f.Truncate(end); err != nil {
				f.Close()
				return nil, err
			}
		}
	}
	a := &Appender{f: f, mu: make(chan struct{}, 1)}
	return a, nil
}

// Append writes one record as a line.
func (a *Appender) Append(v any) error {
	b, err := Marshal(v)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	a.mu <- struct{}{}
	defer func() { <-a.mu }()
	_, err = a.f.Write(b)
	return err
}

// Close closes the file.
func (a *Appender) Close() error { return a.f.Close() }

// DatasetDirs lists data/<ds> directories that have a question.json, sorted.
func DatasetDirs(dataDir string) ([]string, error) {
	ents, err := os.ReadDir(dataDir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		if e.IsDir() {
			if _, err := os.Stat(filepath.Join(dataDir, e.Name(), "question.json")); err == nil {
				out = append(out, e.Name())
			}
		}
	}
	sort.Strings(out)
	return out, nil
}
