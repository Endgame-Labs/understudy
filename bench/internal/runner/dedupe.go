package runner

import (
	"fmt"
	"strings"
	"unicode"
)

// Near-duplicate detection between a training pool and the held-out test set (see separatePool).

// normPrompt normalizes a prompt for duplicate detection: lower case, letters and digits only, single spaces.
func normPrompt(s string) string {
	var b strings.Builder
	space := true
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			space = false
		} else if !space {
			b.WriteByte(' ')
			space = true
		}
	}
	return strings.TrimSpace(b.String())
}

func shingles(norm string) map[string]struct{} {
	w := strings.Fields(norm)
	out := map[string]struct{}{}
	const k = 4
	if len(w) < k {
		return out
	}
	for i := 0; i+k <= len(w); i++ {
		out[strings.Join(w[i:i+k], " ")] = struct{}{}
	}
	return out
}

func jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	if len(a) > len(b) {
		a, b = b, a
	}
	inter := 0
	for k := range a {
		if _, ok := b[k]; ok {
			inter++
		}
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}

// nearDup is the word-4-gram Jaccard similarity from which a (long) prompt counts as a near-duplicate of a test prompt.
const nearDup = 0.8

// nearDupWords is the word-set Jaccard similarity from which a prompt counts as a near-duplicate: it catches small edits
// to short prompts ("How can I build X?" vs "How do I build X?"), which 4-word shingles miss.
const nearDupWords = 0.7

// nearDupEdit is the character edit-distance similarity (1 - distance/longer length, on normalized text) from which a
// prompt counts as a near-duplicate; only checked for prompts up to maxEditLen characters.
const nearDupEdit = 0.85

const maxEditLen = 300

// fingerprint is the precomputed form of a prompt for near-duplicate comparison.
type fingerprint struct {
	norm  string
	sh    map[string]struct{}
	words map[string]struct{}
}

func fingerprintOf(prompt string) fingerprint {
	n := normPrompt(prompt)
	w := map[string]struct{}{}
	for _, f := range strings.Fields(n) {
		w[f] = struct{}{}
	}
	return fingerprint{norm: n, sh: shingles(n), words: w}
}

// editSimilarity is 1 - levenshtein(a, b)/max(len) over runes, 0 when either is too long to be worth comparing.
func editSimilarity(a, b string) float64 {
	ra, rb := []rune(a), []rune(b)
	if len(ra) == 0 || len(rb) == 0 || len(ra) > maxEditLen || len(rb) > maxEditLen {
		return 0
	}
	longer := max(len(ra), len(rb))
	if d := len(ra) - len(rb); d > longer/6+1 || -d > longer/6+1 { // can't reach the threshold
		return 0
	}
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			c := 1
			if ra[i-1] == rb[j-1] {
				c = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+c)
		}
		prev, cur = cur, prev
	}
	return 1 - float64(prev[len(rb)])/float64(longer)
}

// similar reports why two fingerprints are near-duplicates: "" when they are not.
func similar(a, b fingerprint) string {
	if a.norm == b.norm {
		return "same text after normalization"
	}
	if jaccard(a.sh, b.sh) >= nearDup {
		return fmt.Sprintf("word 4-gram Jaccard >= %.2f", nearDup)
	}
	if jaccard(a.words, b.words) >= nearDupWords {
		return fmt.Sprintf("word-set Jaccard >= %.2f", nearDupWords)
	}
	if editSimilarity(a.norm, b.norm) >= nearDupEdit {
		return fmt.Sprintf("edit similarity >= %.2f", nearDupEdit)
	}
	return ""
}

// dedupe holds the held-out (banned) test prompts.
type dedupe struct {
	banned map[string]struct{}
	bfp    []fingerprint
}

// newDedupe returns an empty set.
func newDedupe() *dedupe {
	return &dedupe{banned: map[string]struct{}{}}
}

// Ban adds a held-out test prompt.
func (d *dedupe) Ban(prompt string) {
	n := normPrompt(prompt)
	if _, ok := d.banned[n]; ok {
		return
	}
	d.banned[n] = struct{}{}
	d.bfp = append(d.bfp, fingerprintOf(prompt))
}

// Match reports whether a prompt matches a held-out test prompt (exact or normalized text, or near-duplicate: 4-gram
// shingles, word-set Jaccard or edit similarity). It returns the held-out prompt (normalized) it matched and the reason, both "" when it does not.
func (d *dedupe) Match(prompt string) (string, string) {
	fp := fingerprintOf(prompt)
	if _, ok := d.banned[fp.norm]; ok {
		return fp.norm, "same as a held-out test prompt (exact or normalized text)"
	}
	for _, b := range d.bfp {
		if why := similar(fp, b); why != "" {
			return b.norm, "near-duplicate of a held-out test prompt (" + why + ")"
		}
	}
	return "", ""
}
