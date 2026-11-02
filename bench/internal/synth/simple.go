package synth

import (
	"context"
	"fmt"
	"math/rand/v2"
	"strings"
	"unicode/utf8"
)

// slot is one item a batch asks for: its class (the ground truth) and its spec line for the prompt.
type slot struct {
	Class int
	Truth int
	Spec  string
	Extra map[string]any
}

// batchJob builds a job asking for len(slots) items; mk turns a reply item into a state or a drop reason.
func batchJob(b *Builder, kind string, i int, header string, slots []slot, format string, maxTokens int,
	mk func(s slot, m map[string]any) (any, string)) Job {
	var sb strings.Builder
	sb.WriteString(header)
	sb.WriteString("\n\nWrite these items (one per line below, in order; \"i\" is the item index):\n")
	for j, s := range slots {
		fmt.Fprintf(&sb, "- i=%d: %s\n", j, s.Spec)
	}
	sb.WriteString("\nReply as {\"items\": [")
	sb.WriteString(format)
	sb.WriteString(", ...]} with exactly ")
	fmt.Fprintf(&sb, "%d items, in order.", len(slots))
	return Job{
		Name: fmt.Sprintf("%s/%s/%05d", b.DS.Name, kind, i), System: systemPrompt, User: sb.String(),
		MaxTokens: maxTokens, Planned: len(slots),
		Parse: func(items []map[string]any) []Unit {
			var out []Unit
			for j, m := range ByIndex(items, len(slots)) {
				if m == nil {
					b.Drop("item_missing_or_malformed")
					continue
				}
				st, why := mk(slots[j], m)
				if why != "" {
					b.Drop(why)
					continue
				}
				out = append(out, Unit{Class: slots[j].Class, Members: []Member{{State: st, Truth: slots[j].Truth}}})
			}
			return out
		},
	}
}

// fillSimple is Fill over batchJob with per-slot class assignment by shares.
func fillSimple(ctx context.Context, b *Builder, n, perBatch int, shares []float64,
	plan func(r *rand.Rand, i int, classes []int) Job) ([]Unit, error) {
	return b.Fill(ctx, Quotas(n, shares...), perBatch, func(i int) Job {
		r := b.RNG("batch", i)
		return plan(r, i, Assign(r, perBatch, shares...))
	})
}

// textOK checks a string field's length in runes.
func textOK(s string, lo, hi int) bool {
	n := utf8.RuneCountInString(s)
	return n >= lo && n <= hi
}

// echo checks the item's "label" echo against the assigned label (case-insensitive).
func echo(m map[string]any, want string) bool {
	return strings.EqualFold(Str(m, "label"), want)
}
