package report

import (
	"sort"
	"strconv"
	"strings"

	"github.com/dorkitude/understudy/bench/internal/ds"
)

// ItemMetrics is the item-level view of a per-response dataset (ids "<item>#<k>"): each item's pick is the argmax
// over its N per-response final probabilities of "yes" (what a caller got), compared with the benchmark's correct
// response. Only items whose N sub-questions were all answered locally save a lead call for the whole item (the
// per-response replay counts calls per sub-question).
type ItemMetrics struct {
	Items                int      `json:"items"`
	Responses            int      `json:"responses_per_item"`
	AccLead              float64  `json:"acc_lead"`   // argmax over the lead's per-response p
	AccSystem            float64  `json:"acc_system"` // argmax over the final per-response p
	FullyLocalN          int      `json:"fully_local_n"`
	FullyLocalShare      float64  `json:"fully_local_share"`
	AccLeadLocal         *float64 `json:"acc_lead_on_fully_local"`         // lead-alone item accuracy on the fully-local items
	AccUnderstudentLocal *float64 `json:"acc_understudent_on_fully_local"` // the understudent-only pick (all N answered locally)
	AgreeLocal           *float64 `json:"pick_agreement_on_fully_local"`
	AccSystemNotLoc      *float64 `json:"acc_system_on_other_items"`
	AccLeadNotLoc        *float64 `json:"acc_lead_on_other_items"`
}

// IsPerResponse reports whether every id is "<item>#<k>".
func IsPerResponse(ids []string) bool {
	if len(ids) == 0 {
		return false
	}
	for _, id := range ids {
		if _, _, ok := splitSub(id); !ok {
			return false
		}
	}
	return true
}

func splitSub(id string) (parent string, k int, ok bool) {
	i := strings.LastIndexByte(id, '#')
	if i < 0 {
		return "", 0, false
	}
	k, err := strconv.Atoi(id[i+1:])
	if err != nil {
		return "", 0, false
	}
	return id[:i], k, true
}

// pickYes returns the sub-item with the largest p(yes) (the first on ties); ps[k] holds the probabilities.
func pickYes(ps [][]float64) int {
	best := 0
	for i := range ps {
		if len(ps[i]) > 1 && ps[i][1] > ps[best][1] {
			best = i
		}
	}
	return best
}

// ItemLevel computes item-level metrics from one system's decisions. lead maps ids to the lead's probabilities,
// truth to the per-response ground truth (1 for the correct response). Items with a different number of
// sub-items than the first are rejected by returning nil, as are items without exactly one correct response.
func ItemLevel(dec []ds.Decision, lead map[string][]float64, truth map[string]int) *ItemMetrics {
	type item struct {
		subs []ds.Decision
		ks   []int
	}
	items := map[string]*item{}
	var order []string
	for _, d := range dec {
		p, k, ok := splitSub(d.ID)
		if !ok {
			return nil
		}
		it := items[p]
		if it == nil {
			it = &item{}
			items[p] = it
			order = append(order, p)
		}
		it.subs = append(it.subs, d)
		it.ks = append(it.ks, k)
	}
	m := &ItemMetrics{}
	var tRight, sRight, locN, locT, locS, locAgree, notT, notS int
	for _, p := range order {
		it := items[p]
		idx := make([]int, len(it.subs))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return it.ks[idx[a]] < it.ks[idx[b]] })
		n := len(idx)
		if m.Responses == 0 {
			m.Responses = n
		} else if n != m.Responses {
			return nil
		}
		var tp, sp [][]float64
		correct, ones := -1, 0
		local := true
		for j, i := range idx {
			d := it.subs[i]
			tp = append(tp, lead[d.ID])
			sp = append(sp, d.Probs)
			if truth[d.ID] == 1 {
				correct, ones = j, ones+1
			}
			local = local && d.AnsweredBy == "understudent"
		}
		if ones != 1 {
			return nil
		}
		tPick, sPick := pickYes(tp), pickYes(sp)
		m.Items++
		if tPick == correct {
			tRight++
		}
		if sPick == correct {
			sRight++
		}
		if local {
			locN++
			if tPick == correct {
				locT++
			}
			if sPick == correct {
				locS++
			}
			if tPick == sPick {
				locAgree++
			}
		} else {
			if tPick == correct {
				notT++
			}
			if sPick == correct {
				notS++
			}
		}
	}
	if m.Items == 0 {
		return nil
	}
	f := func(a, b int) *float64 {
		if b == 0 {
			return nil
		}
		return fp(float64(a) / float64(b))
	}
	m.AccLead = float64(tRight) / float64(m.Items)
	m.AccSystem = float64(sRight) / float64(m.Items)
	m.FullyLocalN = locN
	m.FullyLocalShare = float64(locN) / float64(m.Items)
	m.AccLeadLocal, m.AccUnderstudentLocal, m.AgreeLocal = f(locT, locN), f(locS, locN), f(locAgree, locN)
	m.AccLeadNotLoc, m.AccSystemNotLoc = f(notT, m.Items-locN), f(notS, m.Items-locN)
	return m
}

// stubDecisions gives every id a lead-answered decision with no probabilities (to rebuild the lead-only item
// view of a stream).
func stubDecisions(ids []string) []ds.Decision {
	out := make([]ds.Decision, len(ids))
	for i, id := range ids {
		out[i] = ds.Decision{ID: id, AnsweredBy: "lead", Probs: []float64{0, 0}}
	}
	return out
}
