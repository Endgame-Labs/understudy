package report

import (
	"fmt"
	"sort"
	"strings"

	"github.com/endgame-labs/understudy/bench/internal/ds"
	"github.com/endgame-labs/understudy/bench/internal/runner"
)

// LeadRate is what one lead costs per call: uncached input and output dollars per million tokens.
type LeadRate struct {
	InputPerM  float64
	OutputPerM float64
}

// RatesSource says where the rates in LeadRates come from; the report cites it.
const RatesSource = "Fireworks billing API 2026-11-01; TypeSafe list price"

// LeadRates are the measured per-token rates of the leads (the one place they live): Jev and Rune both cost
// $0.042 per million uncached input tokens, and their output is free. otherlead has no measured rate yet.
var LeadRates = map[string]LeadRate{
	"jev":  {InputPerM: 0.042, OutputPerM: 0},
	"rune": {InputPerM: 0.042, OutputPerM: 0},
}

// Tokens estimate for one call: the state's bytes per token, plus the fixed prompt (question instructions and
// criteria, the system One wrapper) that every call carries.
const (
	BytesPerToken = 3.4
	PromptTokens  = 600.0
)

// CallTokens estimates the input tokens of one call from its state's size in bytes.
func CallTokens(stateBytes int) float64 { return float64(stateBytes)/BytesPerToken + PromptTokens }

// CallCost is the dollars one call to lead costs for a state of stateBytes; ok is false for a lead without a
// measured rate.
func CallCost(lead string, stateBytes int) (usd float64, ok bool) {
	r, ok := LeadRates[strings.ToLower(lead)]
	if !ok {
		return 0, false
	}
	return CallTokens(stateBytes) * r.InputPerM / 1e6, true
}

// Usefulness is the per dataset x system view of what the understudent buys: how much it answers, how well it agrees
// with the lead where it answers, how fast it is, and what that saves.
type Usefulness struct {
	N        int     `json:"n"`
	Coverage float64 `json:"coverage"`
	LocalN   int     `json:"local_n"`
	// Agreement with the lead on the answered share, and its Wilson lower bound (z = 1.96).
	Agreement    *float64 `json:"agreement"`
	AgreementLow *float64 `json:"agreement_wilson_low"`
	// UnderstudentUS is the understudent's own time (understudent_us) on the items it answered; ForwardUS on the items it
	// forwarded (the overhead in front of the lead).
	UnderstudentUSP50 *float64 `json:"understudent_us_p50"`
	UnderstudentUSP99 *float64 `json:"understudent_us_p99"`
	ForwardUSP50      *float64 `json:"forward_us_p50"`
	ForwardUSP99      *float64 `json:"forward_us_p99"`
	// Lead latency over the stream's items that carry latency_ms.
	LeadMSP50 *float64 `json:"lead_ms_p50"`
	LeadMSP99 *float64 `json:"lead_ms_p99"`
	// LeadCostPerCall is the mean estimated cost of one lead call over the stream (nil for a lead without a rate);
	// SavedPer1k = Coverage x LeadCostPerCall x 1000; SavedPer1M = Coverage x LeadCostPerCall x 1,000,000.
	LeadCostPerCall *float64 `json:"lead_cost_per_call_usd"`
	SavedPer1k      *float64 `json:"saved_per_1k_calls_usd"`
	SavedPer1M      *float64 `json:"saved_per_1m_calls_usd"`
}

// Percentile is the nearest-rank percentile (p in 0..100) of xs; nil for no values.
func Percentile(xs []float64, p float64) *float64 {
	if len(xs) == 0 {
		return nil
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	rank := int(p/100*float64(len(s)) + 0.999999999)
	if rank < 1 {
		rank = 1
	}
	if rank > len(s) {
		rank = len(s)
	}
	return fp(s[rank-1])
}

// ComputeUsefulness measures dec against the lead's labels (aligned with dec). leadMS are the lead's latencies (ms)
// for the stream, stateBytes the sizes of the stream's states (for the cost), lead the lead's name.
func ComputeUsefulness(dec []ds.Decision, leadLabel []int, leadMS []float64, stateBytes []int, lead string) Usefulness {
	u := Usefulness{N: len(dec)}
	var local, fwd []float64
	agree := 0
	for i, d := range dec {
		isLocal := d.AnsweredBy == "understudent"
		if isLocal {
			u.LocalN++
			if d.Label == leadLabel[i] {
				agree++
			}
		}
		if d.UnderstudentUS != nil {
			if isLocal {
				local = append(local, *d.UnderstudentUS)
			} else {
				fwd = append(fwd, *d.UnderstudentUS)
			}
		}
	}
	if len(dec) > 0 {
		u.Coverage = float64(u.LocalN) / float64(len(dec))
	}
	if u.LocalN > 0 {
		u.Agreement = fp(float64(agree) / float64(u.LocalN))
		lo, _ := runner.Wilson(agree, u.LocalN)
		u.AgreementLow = fp(lo)
	}
	u.UnderstudentUSP50, u.UnderstudentUSP99 = Percentile(local, 50), Percentile(local, 99)
	u.ForwardUSP50, u.ForwardUSP99 = Percentile(fwd, 50), Percentile(fwd, 99)
	u.LeadMSP50, u.LeadMSP99 = Percentile(leadMS, 50), Percentile(leadMS, 99)
	if len(stateBytes) > 0 {
		sum, ok := 0.0, true
		for _, b := range stateBytes {
			c, k := CallCost(lead, b)
			sum += c
			ok = ok && k
		}
		if ok {
			per := sum / float64(len(stateBytes))
			u.LeadCostPerCall = fp(per)
			u.SavedPer1k = fp(u.Coverage * per * 1000)
			u.SavedPer1M = fp(u.Coverage * per * 1000000)
		}
	}
	return u
}

func f1(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("%.1f", *x)
}

func usd(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("$%.6f", *x)
}

// usd2 formats dollars at cent precision (the per-million-calls column).
func usd2(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("$%.2f", *x)
}

// usefulnessMarkdown writes the "Usefulness" section.
func usefulnessMarkdown(b *strings.Builder, r *Results) {
	w := func(f string, a ...any) { fmt.Fprintf(b, f, a...) }
	have := false
	for _, d := range r.Datasets {
		for _, s := range d.Systems {
			have = have || s.Usefulness != nil
		}
	}
	if !have {
		return
	}
	w("\n## Usefulness\n\n")
	w("Per dataset × system, over every item of the run. **coverage** = share answered by the understudent; **agree** = agreement with "+
		"the lead on that share, with its Wilson lower bound (z = 1.96); **us** = the understudent's own time (`understudent_us`) p50/p99 on the "+
		"items it answered, **fwd us** on the items it forwarded (the overhead before the lead); **lead ms** = the lead's recorded "+
		"latency p50/p99 (`latency_ms`); **$ saved / 1M calls** = coverage × lead cost per call × 1,000,000. Lead cost per call = "+
		"(state bytes / %.1f + %.0f) tokens × the rate of the lead (Jev and Rune: $%.3f per million uncached input tokens, output free; "+
		"%s), averaged over the stream's states. "+
		"Savings scale linearly with call volume; the understudent's own cost is CPU time (see docs/benchmarks.md#cost-at-scale).\n\n", BytesPerToken, PromptTokens, LeadRates["jev"].InputPerM, RatesSource)
	w("| dataset | system | coverage | agree (Wilson low) | us p50/p99 | fwd us p50/p99 | lead ms p50/p99 | lead $/call | $ saved / 1M calls |\n|---|---|---|---|---|---|---|---|---|\n")
	for _, d := range r.Datasets {
		for _, s := range d.Systems {
			u := s.Usefulness
			if u == nil {
				continue
			}
			ag := "–"
			if u.Agreement != nil {
				ag = fmt.Sprintf("%s (%s) n=%d", pct(*u.Agreement), pct(*u.AgreementLow), u.LocalN)
			}
			w("| %s | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s | %s |\n", d.Name, s.System, pct(u.Coverage), ag,
				usStr(u.UnderstudentUSP50), usStr(u.UnderstudentUSP99), usStr(u.ForwardUSP50), usStr(u.ForwardUSP99),
				f1(u.LeadMSP50), f1(u.LeadMSP99), usd(u.LeadCostPerCall), usd2(u.SavedPer1M))
		}
	}
}
