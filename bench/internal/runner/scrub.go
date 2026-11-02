package runner

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/endgame-labs/understudy/bench/internal/ds"
)

// DataPath resolves a dataset directory: dataDir is relative to the repository root unless absolute.
func DataPath(root, dataDir, name string) string {
	if dataDir == "" {
		dataDir = "data"
	}
	if !filepath.IsAbs(dataDir) {
		dataDir = filepath.Join(root, dataDir)
	}
	return filepath.Join(dataDir, name)
}

// IsPrivateDataDir is true for the data-private directory (benchmark and Enron text that must not be committed).
func IsPrivateDataDir(dataDir string) bool {
	return filepath.Base(filepath.Clean(dataDir)) == "data-private"
}

// StreamIDsFile holds a stream's ids and lead probabilities without the states; the report reads it when
// stream.jsonl has been removed.
const StreamIDsFile = "stream.ids.jsonl"

// idRe accepts dataset item ids, including per-response ids "<item>#<k>".
var idRe = regexp.MustCompile(`^[A-Za-z0-9_.:#-]{1,64}$`)

// Scrub removes every state-bearing artifact of a dataset's run directory (stream.jsonl, train.jsonl, operate.jsonl,
// snapshot directories), keeping StreamIDsFile (ids and lead probabilities), the decisions, timings and summaries,
// and then verifies that nothing that remains carries state text.
func Scrub(outDir, itemsPath string) error {
	stream := filepath.Join(outDir, "stream.jsonl")
	if _, err := os.Stat(stream); err == nil {
		rows, err := ds.ReadJSONL[ds.StreamItem](stream)
		if err != nil {
			return err
		}
		for i := range rows {
			rows[i].State = nil
		}
		type idRow struct {
			ID        string    `json:"id"`
			LeadProbs []float64 `json:"lead_probs"`
		}
		ids := make([]idRow, len(rows))
		for i, r := range rows {
			ids[i] = idRow{r.ID, r.LeadProbs}
		}
		if err := ds.WriteJSONL(filepath.Join(outDir, StreamIDsFile), ids); err != nil {
			return err
		}
	}
	for _, f := range []string{"stream.jsonl", "train.jsonl", "operate.jsonl"} {
		if err := os.Remove(filepath.Join(outDir, f)); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	snaps, _ := filepath.Glob(filepath.Join(outDir, "snapshot.*"))
	for _, p := range snaps {
		if err := os.RemoveAll(p); err != nil {
			return err
		}
	}
	if err := StripChecklists(outDir); err != nil {
		return err
	}
	return VerifyNoText(outDir, itemsPath)
}

// checklistListingRe matches a line of a checklist's rendered listing ("3. IF word:cat THEN class 1 ...").
var checklistListingRe = regexp.MustCompile(`\d+\. IF \S`)

// StripChecklists removes the induced checklist rules and listings (report.checklist) from a dataset's summaries and
// stderr logs. The rules are n-grams of the training text, so they are as sensitive as the text itself.
func StripChecklists(outDir string) error {
	sums, _ := filepath.Glob(filepath.Join(outDir, "summary.*.json"))
	for _, p := range sums {
		if err := StripChecklistsFile(p); err != nil {
			return err
		}
	}
	logs, _ := filepath.Glob(filepath.Join(outDir, "*.log"))
	for _, p := range logs {
		if err := stripChecklistLog(p); err != nil {
			return err
		}
	}
	return nil
}

// StripChecklistsFile deletes the "rules" and "listing" keys of every "checklist" object in a JSON file (a summary or
// results.json). A file with no such object is left untouched.
func StripChecklistsFile(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if !bytes.Contains(b, []byte(`"checklist"`)) {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if !dropChecklistRules(v) {
		return nil
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return err
	}
	return os.WriteFile(path, out.Bytes(), 0o644)
}

func dropChecklistRules(v any) (changed bool) {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			if m, ok := e.(map[string]any); ok && k == "checklist" {
				for _, key := range []string{"rules", "listing"} {
					if _, has := m[key]; has {
						delete(m, key)
						changed = true
					}
				}
			}
			if dropChecklistRules(e) {
				changed = true
			}
		}
	case []any:
		for _, e := range x {
			if dropChecklistRules(e) {
				changed = true
			}
		}
	}
	return changed
}

// stripChecklistLog removes the "rules" and "listing" entries of the pretty-printed "checklist" objects a log echoes,
// and any stray listing line.
func stripChecklistLog(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	lines := strings.Split(string(b), "\n")
	var out []string
	inList := false // inside a "checklist": { ... } object
	skipUntil := "" // closing prefix of a "rules": [ ... ] block being skipped
	indent := func(l string) string { return l[:len(l)-len(strings.TrimLeft(l, " "))] }
	changed := false
	var chIndent string
	for _, l := range lines {
		t := strings.TrimSpace(l)
		switch {
		case skipUntil != "":
			if strings.HasPrefix(l, skipUntil) && strings.HasPrefix(t, "]") {
				skipUntil = ""
			}
			continue
		case inList && strings.HasPrefix(l, chIndent+"  ") && strings.HasPrefix(t, `"rules": [`):
			if !strings.HasSuffix(strings.TrimRight(t, ","), "]") {
				skipUntil = chIndent + "  "
			}
			changed = true
			continue
		case inList && strings.HasPrefix(t, `"listing":`):
			changed = true
			continue
		case checklistListingRe.MatchString(t) && strings.Contains(t, " THEN "):
			changed = true
			continue
		}
		if strings.HasPrefix(t, `"checklist": {`) && !strings.HasSuffix(t, "}") && !strings.HasSuffix(t, "},") {
			inList, chIndent = true, indent(l)
		} else if inList && strings.HasPrefix(l, chIndent+"}") {
			inList = false
		}
		out = append(out, l)
	}
	if !changed {
		return nil
	}
	return os.WriteFile(path, []byte(strings.Join(out, "\n")), 0o644)
}

// VerifyNoText checks a dataset's run directory: decisions files must parse strictly as decisions (only the SPEC
// fields) with short ids and mode/readout names, and no other remaining file may contain a verbatim 32-rune head or
// tail of any state string of the dataset's items.
func VerifyNoText(outDir, itemsPath string) error {
	ents, err := os.ReadDir(outDir)
	if err != nil {
		return err
	}
	var others []string
	for _, e := range ents {
		n := e.Name()
		p := filepath.Join(outDir, n)
		switch {
		case e.IsDir():
			return fmt.Errorf("%s: unexpected directory %s in a text-free run", outDir, n)
		case n == "stream.jsonl" || n == "train.jsonl" || n == "operate.jsonl":
			return fmt.Errorf("%s: %s still present", outDir, n)
		case strings.HasPrefix(n, "decisions.") && strings.HasSuffix(n, ".jsonl"):
			if err := verifyDecisions(p); err != nil {
				return err
			}
		case n == StreamIDsFile:
			if err := verifyStreamIDs(p); err != nil {
				return err
			}
		default:
			others = append(others, p)
		}
	}
	for _, p := range others {
		if b, err := os.ReadFile(p); err == nil && checklistListingRe.Match(b) && bytes.Contains(b, []byte(" THEN ")) {
			return fmt.Errorf("%s still carries checklist rules", p)
		}
	}
	return ScanFiles(others, itemsPath)
}

// ScanFiles fails if any file contains a verbatim 32-rune head or tail of a state string of the items.
func ScanFiles(others []string, itemsPath string) error {
	if len(others) == 0 {
		return nil
	}
	var needles [][]byte
	items, err := ds.ReadJSONL[ds.Item](itemsPath)
	if err != nil {
		return err
	}
	for _, it := range items {
		var v any
		if json.Unmarshal(it.State, &v) != nil {
			continue
		}
		collectNeedles(v, &needles)
	}
	for _, p := range others {
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		for _, nd := range needles {
			if bytes.Contains(b, nd) {
				return fmt.Errorf("%s contains state text (%q...)", p, truncRunes(string(nd), 20))
			}
		}
	}
	return nil
}

func collectNeedles(v any, out *[][]byte) {
	switch x := v.(type) {
	case string:
		if utf8.RuneCountInString(x) >= 40 {
			r := []rune(x)
			*out = append(*out, jsonFragment(string(r[:32])), jsonFragment(string(r[len(r)-32:])))
		}
	case []any:
		for _, e := range x {
			collectNeedles(e, out)
		}
	case map[string]any:
		for _, e := range x {
			collectNeedles(e, out)
		}
	}
}

// jsonFragment returns s as it would appear inside a JSON string (and raw), whichever the file used: the raw text
// when it has no characters JSON escapes, else the escaped form without its quotes.
func jsonFragment(s string) []byte {
	b, _ := ds.Marshal(s)
	e := b[1 : len(b)-1]
	if bytes.Contains(e, []byte("\\")) {
		return e
	}
	return []byte(s)
}

func truncRunes(s string, n int) string {
	r := []rune(s)
	if len(r) > n {
		r = r[:n]
	}
	return string(r)
}

func verifyDecisions(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	for line := 1; ; line++ {
		var d ds.Decision
		if err := dec.Decode(&d); err == io.EOF {
			return nil
		} else if err != nil {
			return fmt.Errorf("%s: decision %d is not a plain decision record: %w", path, line, err)
		}
		if !idRe.MatchString(d.ID) {
			return fmt.Errorf("%s: decision %d has a suspicious id (%d bytes)", path, line, len(d.ID))
		}
		for _, s := range []*string{d.Mode, d.Readout} {
			if s != nil && len(*s) > 32 {
				return fmt.Errorf("%s: decision %d has a long mode/readout string", path, line)
			}
		}
		if len(d.AnsweredBy) > 16 {
			return fmt.Errorf("%s: decision %d has a long answered_by", path, line)
		}
	}
}

func verifyStreamIDs(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	for line := 1; ; line++ {
		var d struct {
			ID        string    `json:"id"`
			LeadProbs []float64 `json:"lead_probs"`
		}
		if err := dec.Decode(&d); err == io.EOF {
			return nil
		} else if err != nil {
			return fmt.Errorf("%s: line %d: %w", path, line, err)
		}
		if !idRe.MatchString(d.ID) {
			return fmt.Errorf("%s: line %d has a suspicious id", path, line)
		}
	}
}
