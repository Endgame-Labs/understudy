package report

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/endgame-labs/understudy/bench/internal/ds"
	"github.com/endgame-labs/understudy/bench/internal/runner"
)

// Expected is the predicted outcome per category (question.json notes say why).
var Expected = map[string]map[string]string{
	"surface_property":  {"*": "takes over"},
	"topic_kind":        {"*": "takes over"},
	"graded_tone":       {"*": "partial"},
	"needle":            {"understudent": "takes over", "hunch": "abstains or partial"},
	"minimal_pair":      {"*": "partial or abstains"},
	"pair_compare":      {"understudent": "takes over", "hunch": "abstains"},
	"relational_match":  {"*": "abstains"},
	"binding_lookup":    {"*": "abstains"},
	"conditional_rule":  {"*": "abstains"},
	"world_knowledge":   {"*": "abstains"},
	"compound_judgment": {"*": "abstains"},
}

func expected(category, system string) string {
	m := Expected[category]
	base := strings.TrimSuffix(system, "-split")
	if v, ok := m[base]; ok {
		return v
	}
	if v, ok := m["*"]; ok {
		return v
	}
	return "?"
}

// LeadQuality is the lead measured against the generator's labels.
type LeadQuality struct {
	N           int       `json:"n"`
	Accuracy    float64   `json:"accuracy"`
	Recall      []float64 `json:"recall_per_class"`
	Predicted   []int     `json:"predicted_counts"`
	Flag        string    `json:"flag,omitempty"`
	MeanLatency float64   `json:"mean_latency_ms"`
}

// SystemResult is one system's metrics on one dataset.
type SystemResult struct {
	System  string   `json:"system"`
	Mode    string   `json:"mode"`
	Metrics *Metrics `json:"metrics,omitempty"`
	// OperateSlice: for cold-start systems in a run with split mode, the same metrics over the operate items only
	OperateSlice *Metrics       `json:"operate_slice,omitempty"`
	Timing       *runner.Timing `json:"timing,omitempty"`
	Expected     string         `json:"expected"`
	Missing      string         `json:"missing,omitempty"`
	// CoverageByReadout comes from the system's stderr summary (summary.<system>.json), e.g. a ladder's tiers.
	CoverageByReadout map[string]any `json:"coverage_by_readout,omitempty"`
	// CV is the understudent's own cross-validated agreement (from the summary), against the majority share.
	CV *CVInfo `json:"cv,omitempty"`
	// Items is the item-level view of a per-response dataset (ids "<item>#<k>").
	Items *ItemMetrics `json:"items,omitempty"`
	// Usefulness: coverage, agreement with its Wilson bound, latency and dollars saved (see ComputeUsefulness).
	Usefulness *Usefulness `json:"usefulness,omitempty"`
}

// CVInfo is a system's cross-validation in its summary: the best CV agreement over all refits and the final one,
// with the majority-class share of the labelled sample.
type CVInfo struct {
	BestAgreement  *float64 `json:"best_agreement"`
	FinalAgreement *float64 `json:"final_agreement"`
	MajorityShare  *float64 `json:"majority_share"`
}

func readCV(path string) *CVInfo {
	var sum struct {
		Refits []struct {
			CV *float64 `json:"cv_agreement"`
		} `json:"refits"`
		Final struct {
			CV     *float64 `json:"cv_agreement"`
			Report struct {
				Maj *float64 `json:"majority_share"`
			} `json:"report"`
		} `json:"final"`
		FinalReport struct {
			CV  *float64 `json:"cv_agreement"`
			Maj *float64 `json:"majority_share"`
		} `json:"final_report"`
	}
	if ds.ReadJSON(path, &sum) != nil {
		return nil
	}
	cv := &CVInfo{FinalAgreement: sum.Final.CV, MajorityShare: sum.Final.Report.Maj}
	if sum.FinalReport.CV != nil {
		cv.FinalAgreement, cv.MajorityShare = sum.FinalReport.CV, sum.FinalReport.Maj
	}
	for _, r := range sum.Refits {
		if r.CV != nil && (cv.BestAgreement == nil || *r.CV > *cv.BestAgreement) {
			cv.BestAgreement = r.CV
		}
	}
	if cv.BestAgreement == nil {
		cv.BestAgreement = cv.FinalAgreement
	}
	if cv.BestAgreement == nil && cv.MajorityShare == nil {
		return nil
	}
	return cv
}

// DatasetResult is one dataset.
type DatasetResult struct {
	Name        string         `json:"name"`
	Category    string         `json:"category"`
	Type        string         `json:"type"`
	Options     []string       `json:"options"`
	N           int            `json:"n"`
	ClassCounts []int          `json:"class_counts"`
	Lead        LeadQuality    `json:"lead"`
	Systems     []SystemResult `json:"systems"`
	// PerResponse marks a per-response dataset (ids "<item>#<k>"); Items is then the number of parent items.
	PerResponse bool `json:"per_response,omitempty"`
	Items       int  `json:"items,omitempty"`
	// LeadItemAcc is the lead's item-level accuracy (argmax over an item's per-response p).
	LeadItemAcc *float64 `json:"lead_item_acc,omitempty"`
}

// Results is runs/<run>/results.json.
type Results struct {
	Run       string          `json:"run"`
	Lead      string          `json:"lead"`
	Train     int             `json:"train"`
	Generated string          `json:"generated"`
	Datasets  []DatasetResult `json:"datasets"`
}

// Build computes the results of a run.
func Build(root, run, dataDir string) (*Results, error) {
	runDir := filepath.Join(root, "runs", run)
	var meta runner.RunMeta
	if err := ds.ReadJSON(filepath.Join(runDir, "run.json"), &meta); err != nil {
		return nil, err
	}
	if dataDir != "" {
		meta.DataDir = dataDir
	}
	res := &Results{Run: run, Lead: meta.Lead, Train: meta.Train, Generated: time.Now().UTC().Format(time.RFC3339)}
	split := false
	for _, m := range meta.Modes {
		split = split || m == "split"
	}
	for _, name := range meta.Datasets {
		dr, err := buildDataset(root, runDir, name, meta, split)
		if err != nil {
			return nil, err
		}
		res.Datasets = append(res.Datasets, *dr)
	}
	return res, nil
}

func buildDataset(root, runDir, name string, meta runner.RunMeta, split bool) (*DatasetResult, error) {
	dataPath := runner.DataPath(root, meta.DataDir, name)
	dir := filepath.Join(runDir, name)
	q, err := ds.ReadQuestion(filepath.Join(dir, "question.json"))
	if err != nil {
		return nil, err
	}
	items, err := ds.ReadJSONL[ds.Item](filepath.Join(dataPath, "items.jsonl"))
	if err != nil {
		return nil, err
	}
	truth := map[string]int{}
	for _, it := range items {
		truth[it.ID] = it.Truth
	}
	stream, err := ds.ReadJSONL[ds.StreamItem](filepath.Join(dir, "stream.jsonl"))
	if err != nil {
		return nil, err
	}
	if len(stream) == 0 { // a text-free run keeps only the ids and lead probabilities
		if stream, err = ds.ReadJSONL[ds.StreamItem](filepath.Join(dir, runner.StreamIDsFile)); err != nil {
			return nil, err
		}
	}
	recs, err := ds.ReadJSONL[ds.LeadRecord](ds.LeadPath(dataPath, meta.Lead))
	if err != nil {
		return nil, err
	}
	k := q.K()
	tl := map[string]int{}
	var tlabels, tTruth []int
	for _, s := range stream {
		l := ds.Argmax(s.LeadProbs)
		tl[s.ID] = l
		tlabels = append(tlabels, l)
		tTruth = append(tTruth, truth[s.ID])
	}
	dr := &DatasetResult{Name: name, Category: q.Category, Type: q.Type, Options: q.Options, N: len(stream),
		ClassCounts: make([]int, k)}
	for _, y := range tTruth {
		dr.ClassCounts[y]++
	}
	dr.Lead = leadQuality(tlabels, tTruth, k, recs)
	majority := Majority(tlabels, k)
	pos := map[string]int{}
	tprobs := map[string][]float64{}
	var ids []string
	for i, s := range stream {
		pos[s.ID] = i
		tprobs[s.ID] = s.LeadProbs
		ids = append(ids, s.ID)
	}
	// the lead's latency and the states' sizes, for the usefulness view
	recByID := map[string]ds.LeadRecord{}
	for _, r := range recs {
		recByID[r.ID] = r
	}
	sizeByID := map[string]int{}
	for _, it := range items {
		sizeByID[it.ID] = len(it.State)
	}
	var sizes []int
	for _, id := range ids {
		sizes = append(sizes, sizeByID[id])
	}
	perResp := q.Type == ds.Noul && IsPerResponse(ids)
	if perResp {
		dr.PerResponse = true
		if im := ItemLevel(stubDecisions(ids), tprobs, truth); im != nil {
			dr.Items = im.Items
			dr.LeadItemAcc = fp(im.AccLead)
		}
	}
	for _, mode := range meta.Modes {
		for _, sys := range meta.Systems {
			sn := runner.SystemName(sys, mode)
			sr := SystemResult{System: sn, Mode: mode, Expected: expected(q.Category, sn)}
			var t runner.Timing
			if ds.ReadJSON(filepath.Join(dir, "timing."+sn+".json"), &t) == nil {
				sr.Timing = &t
			}
			var sum struct {
				CoverageByReadout map[string]any `json:"coverage_by_readout"`
			}
			if ds.ReadJSON(filepath.Join(dir, "summary."+sn+".json"), &sum) == nil && len(sum.CoverageByReadout) > 0 {
				sr.CoverageByReadout = sum.CoverageByReadout
			}
			sr.CV = readCV(filepath.Join(dir, "summary."+sn+".json"))
			dec, err := ds.ReadDecisions(filepath.Join(dir, "decisions."+sn+".jsonl"))
			if err != nil {
				return nil, fmt.Errorf("%s %s: %w", name, sn, err)
			}
			want := len(stream)
			if mode == "split" {
				want = len(stream) - meta.Train
			}
			if len(dec) == 0 || len(dec) != want {
				sr.Missing = fmt.Sprintf("%d decisions for %d items", len(dec), want)
				if sr.Timing != nil && sr.Timing.Skipped != "" {
					sr.Missing = sr.Timing.Skipped
				}
				dr.Systems = append(dr.Systems, sr)
				continue
			}
			tt, yy, bad := make([]int, len(dec)), make([]int, len(dec)), ""
			for i, d := range dec {
				p, ok := pos[d.ID]
				if !ok {
					bad = "unknown id " + d.ID
					break
				}
				if mode == "replay" && p != i || mode == "split" && p != i+meta.Train {
					bad = "decisions out of stream order at " + d.ID
					break
				}
				if d.Label < 0 || d.Label >= k {
					bad = "label out of range at " + d.ID
					break
				}
				tt[i], yy[i] = tl[d.ID], truth[d.ID]
			}
			if bad != "" {
				sr.Missing = bad
				dr.Systems = append(dr.Systems, sr)
				continue
			}
			m := Compute(dec, tt, yy, k, majority)
			sr.Metrics = &m
			{
				off := 0 // split mode judges the items after the seed answers
				if mode == "split" {
					off = meta.Train
				}
				u := ComputeUsefulness(dec, tt, LeadLatencies(ids[off:], recByID), sizes[off:], meta.Lead)
				sr.Usefulness = &u
			}
			if perResp && mode == "replay" {
				sr.Items = ItemLevel(dec, tprobs, truth)
			}
			if mode == "replay" && split && meta.Train < len(dec) {
				o := Compute(dec[meta.Train:], tt[meta.Train:], yy[meta.Train:], k, majority)
				sr.OperateSlice = &o
			}
			dr.Systems = append(dr.Systems, sr)
		}
	}
	return dr, nil
}

// LeadLatencies are the lead's recorded latencies (ms) for exactly the given ids (the evaluated ones, the population
// that coverage, agreement and understudent latency describe), skipping ids with no positive latency.
func LeadLatencies(ids []string, recByID map[string]ds.LeadRecord) []float64 {
	var out []float64
	for _, id := range ids {
		if r, ok := recByID[id]; ok && r.LatencyMS > 0 {
			out = append(out, float64(r.LatencyMS))
		}
	}
	return out
}

func leadQuality(pred, truth []int, k int, recs []ds.LeadRecord) LeadQuality {
	tq := LeadQuality{N: len(pred), Recall: make([]float64, k), Predicted: make([]int, k)}
	right := 0
	per, perN := make([]int, k), make([]int, k)
	for i := range pred {
		tq.Predicted[pred[i]]++
		perN[truth[i]]++
		if pred[i] == truth[i] {
			right++
			per[truth[i]]++
		}
	}
	if len(pred) > 0 {
		tq.Accuracy = float64(right) / float64(len(pred))
	}
	for c := range per {
		if perN[c] > 0 {
			tq.Recall[c] = round(float64(per[c])/float64(perN[c]), 4)
		}
	}
	var lat float64
	for _, r := range recs {
		lat += float64(r.LatencyMS)
	}
	if len(recs) > 0 {
		tq.MeanLatency = round(lat/float64(len(recs)), 1)
	}
	switch {
	case tq.Accuracy < 0.70:
		tq.Flag = "SUSPECT: lead far below the generator's labels; the labels or the lead are wrong"
	case tq.Accuracy < 0.85:
		tq.Flag = "check: lead well below the generator's labels"
	}
	return tq
}

// ---------------------------------------------------------------- markdown

func pct(x float64) string { return fmt.Sprintf("%.1f%%", 100*x) }

func pctp(x *float64) string {
	if x == nil {
		return "–"
	}
	return pct(*x)
}

func f2p(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("%.2f", *x)
}

func yn(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// Markdown renders REPORT.md.
func Markdown(r *Results) string {
	var b strings.Builder
	w := func(f string, a ...any) { fmt.Fprintf(&b, f, a...) }
	w("# understudy-bench run `%s`\n\n", r.Run)
	w("Lead: **%s**. Generated %s.\n\n", r.Lead, r.Generated)
	w("Systems: `understudent` / `hunch` = cold start (replay of the whole stream from an empty store)")
	if r.Train > 0 {
		w("; `*-split` = warm start (seeded with %d lead answers via train, then judging the remaining items; "+
			"its positions, halves, accuracies and calls are over those operate items only, not counting the %d seed answers)", r.Train, r.Train)
	}
	w(".\n\n")
	w("Verdicts: **takes over** = 2nd-half coverage ≥ 50%% and accuracy within 1 pt of the lead alone; " +
		"**harmful** = graduated and accuracy > 2 pts below the lead; **correctly abstains** = never graduated; " +
		"otherwise **partial**. Accuracy is against the generator's labels (ground truth by construction). " +
		"Agreement, kappa and minority recall are against the lead, on the items the understudent answered.\n\n")

	w("## Lead vs ground truth\n\n")
	w("| dataset | category | type | n | class counts | lead acc | recall per class | flag |\n|---|---|---|---|---|---|---|---|\n")
	for _, d := range r.Datasets {
		var rec []string
		for _, x := range d.Lead.Recall {
			rec = append(rec, fmt.Sprintf("%.2f", x))
		}
		w("| %s | %s | %s | %d | %v | %s | %s | %s |\n", d.Name, d.Category, d.Type, d.N, d.ClassCounts,
			pct(d.Lead.Accuracy), strings.Join(rec, " / "), d.Lead.Flag)
	}

	byCat := map[string][]DatasetResult{}
	var cats []string
	for _, d := range r.Datasets {
		if _, ok := byCat[d.Category]; !ok {
			cats = append(cats, d.Category)
		}
		byCat[d.Category] = append(byCat[d.Category], d)
	}
	w("\n## Results by category\n\n")
	w("| category | dataset | type | system | grad. | first local | coverage | cov. 2nd half | local agree | audit agree | majority base | kappa | minority recall | acc lead | acc system | acc understudent | calls saved | µs/item | verdict | expected |\n")
	w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, c := range cats {
		for _, d := range byCat[c] {
			for _, s := range d.Systems {
				if s.Metrics == nil {
					w("| %s | %s | %s | %s | | | | | | | | | | | | | | | missing: %s | %s |\n", c, d.Name, d.Type, s.System, s.Missing, s.Expected)
					continue
				}
				m := s.Metrics
				first := "–"
				if m.FirstUnderstudentItem != nil {
					first = fmt.Sprint(*m.FirstUnderstudentItem)
				}
				us := "–"
				if m.UnderstudentUSMean != nil {
					us = fmt.Sprintf("%.0f", *m.UnderstudentUSMean)
				}
				w("| %s | %s | %s | %s | %s | %s | %s | %s | %s (%d) | %s (%d) | %s | %s | %s (%d) | %s | %s | %s | %d/%d | %s | **%s** | %s |\n",
					c, d.Name, d.Type, s.System, yn(m.Graduated), first, pct(m.Coverage), pct(m.Coverage2ndHalf),
					pctp(m.LocalAgreement), m.LocalN, pctp(m.AuditAgreement), m.AuditN, pctp(m.MajorityBaseline), f2p(m.Kappa),
					pctp(m.LocalMinorityRecal), m.LocalMinorityN, pct(m.AccLead), pct(m.AccSystem), pctp(m.AccUnderstudent),
					m.CallsSaved, m.N, us, m.Verdict, s.Expected)
			}
		}
	}

	for _, mode := range []string{"replay", "split"} {
		seen := false
		for _, d := range r.Datasets {
			for _, s := range d.Systems {
				seen = seen || s.Mode == mode
			}
		}
		if !seen {
			continue
		}
		title := "cold start"
		if mode == "split" {
			title = "warm start, operate items"
		}
		w("\n## Winners by category (%s)\n\n", title)
		w("Winner = a system that graduates with accuracy within 1 pt of the lead alone, ranked by 2nd-half coverage, ties to the lower µs/item.\n\n")
		w("| category | dataset | type | winner | cov. 2nd half | acc system | acc lead | µs/item | runner-up |\n|---|---|---|---|---|---|---|---|---|\n")
		for _, c := range cats {
			for _, d := range byCat[c] {
				ws := Winners(d, mode)
				if len(ws) == 0 {
					w("| %s | %s | %s | none | | | | | |\n", c, d.Name, d.Type)
					continue
				}
				m := ws[0].Metrics
				up := "–"
				if len(ws) > 1 {
					up = fmt.Sprintf("%s (%s)", ws[1].System, pct(ws[1].Metrics.Coverage2ndHalf))
				}
				w("| %s | %s | %s | **%s** | %s | %s | %s | %s | %s |\n", c, d.Name, d.Type, ws[0].System, pct(m.Coverage2ndHalf),
					pct(m.AccSystem), pct(m.AccLead), usStr(m.UnderstudentUSMean), up)
			}
		}
	}

	perRespMarkdown(&b, r)

	hasTiers := false
	for _, d := range r.Datasets {
		for _, s := range d.Systems {
			hasTiers = hasTiers || len(s.CoverageByReadout) > 0
		}
	}
	if hasTiers {
		w("\n## Coverage by readout (from each system's summary)\n\n| dataset | system | coverage_by_readout |\n|---|---|---|\n")
		for _, d := range r.Datasets {
			for _, s := range d.Systems {
				if len(s.CoverageByReadout) == 0 {
					continue
				}
				keys := make([]string, 0, len(s.CoverageByReadout))
				for k := range s.CoverageByReadout {
					keys = append(keys, k)
				}
				sort.Strings(keys)
				var parts []string
				for _, k := range keys {
					v := s.CoverageByReadout[k]
					if f, ok := v.(float64); ok && f <= 1 {
						parts = append(parts, fmt.Sprintf("%s %s", k, pct(f)))
					} else {
						parts = append(parts, fmt.Sprintf("%s %v", k, v))
					}
				}
				w("| %s | %s | %s |\n", d.Name, s.System, strings.Join(parts, ", "))
			}
		}
	}

	w("\n## Summary by wire type\n\n")
	w("| type | system | datasets | takes over | partial | correctly abstains | harmful | mean cov. 2nd half | mean Δacc vs lead |\n|---|---|---|---|---|---|---|---|---|\n")
	type agg struct {
		n, take, part, abst, harm int
		cov, dacc                 float64
	}
	keys := map[string]*agg{}
	var order []string
	for _, d := range r.Datasets {
		for _, s := range d.Systems {
			if s.Metrics == nil {
				continue
			}
			key := d.Type + "\x00" + s.System
			a, ok := keys[key]
			if !ok {
				a = &agg{}
				keys[key] = a
				order = append(order, key)
			}
			a.n++
			switch s.Metrics.Verdict {
			case "takes over":
				a.take++
			case "partial":
				a.part++
			case "correctly abstains":
				a.abst++
			case "harmful":
				a.harm++
			}
			a.cov += s.Metrics.Coverage2ndHalf
			a.dacc += s.Metrics.AccSystem - s.Metrics.AccLead
		}
	}
	sort.Strings(order)
	for _, key := range order {
		a := keys[key]
		p := strings.SplitN(key, "\x00", 2)
		w("| %s | %s | %d | %d | %d | %d | %d | %s | %+.1f pt |\n", p[0], p[1], a.n, a.take, a.part, a.abst, a.harm,
			pct(a.cov/float64(a.n)), 100*a.dacc/float64(a.n))
	}

	if r.Train > 0 {
		w("\n## Warm start vs cold start (operate items only: stream items %d..end)\n\n", r.Train+1)
		w("Warm start = seeded with %d lead answers via train, then judging. Cold start = the replay's decisions on the same items.\n\n", r.Train)
		w("| dataset | category | system | graduated right after training | coverage | cov. 2nd half | acc lead | acc system | kappa | verdict |\n|---|---|---|---|---|---|---|---|---|---|\n")
		for _, d := range r.Datasets {
			for _, s := range d.Systems {
				m := s.Metrics
				label := s.System
				grad := "–"
				if s.Mode == "replay" {
					m = s.OperateSlice
					label += " (cold, same items)"
				} else if m != nil {
					grad = yn(m.GraduatedAtStart)
				}
				if m == nil {
					if s.Mode == "split" {
						w("| %s | %s | %s | | | | | | | missing: %s |\n", d.Name, d.Category, label, s.Missing)
					}
					continue
				}
				w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", d.Name, d.Category, label, grad, pct(m.Coverage),
					pct(m.Coverage2ndHalf), pct(m.AccLead), pct(m.AccSystem), f2p(m.Kappa), m.Verdict)
			}
		}
	}

	usefulnessMarkdown(&b, r)

	w("\n## Timing\n\n| dataset | system | wall s | train s | items | status |\n|---|---|---|---|---|---|\n")
	for _, d := range r.Datasets {
		for _, s := range d.Systems {
			if s.Timing == nil {
				continue
			}
			t := s.Timing
			tr := "–"
			if t.TrainWallS != nil {
				tr = fmt.Sprintf("%.1f", *t.TrainWallS)
			}
			st := "ok"
			if t.Skipped != "" {
				st = "skipped: " + t.Skipped
			} else if t.Error != "" {
				st = "error: " + t.Error
			}
			w("| %s | %s | %.1f | %s | %d | %s |\n", d.Name, s.System, t.WallS, tr, t.Items, st)
		}
	}
	w("\nNotes: `audit agree` covers local answers the lead audited and forwarded items the understudent would have " +
		"answered (eligible). hunch fields that are approximations are listed in the docstring of bench/hunch_runner/run_hunch.py.\n")
	return b.String()
}

// Write builds and writes results.json and REPORT.md.
func Write(root, run, dataDir string) (*Results, error) {
	var lm runner.RunMeta
	if ds.ReadJSON(filepath.Join(root, "runs", run, "run.json"), &lm) == nil && len(lm.Ladder) > 0 {
		// a fixed-test learning-curve run has its own results (LADDER.md); there are no per-mode decisions to score
		if dataDir != "" {
			lm.DataDir = dataDir
		}
		lr, err := writeLadder(root, run, dataDir, lm)
		if lr != nil {
			return &Results{Run: run, Lead: lm.Lead, Generated: lr.Generated}, err
		}
		return nil, err
	}
	r, err := Build(root, run, dataDir)
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(root, "runs", run)
	if err := ds.WriteJSON(filepath.Join(dir, "results.json"), r); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(dir, "REPORT.md"), []byte(Markdown(r)), 0o644); err != nil {
		return nil, err
	}
	var meta runner.RunMeta
	if err := ds.ReadJSON(filepath.Join(dir, "run.json"), &meta); err != nil {
		return nil, err
	}
	if dataDir != "" {
		meta.DataDir = dataDir
	}
	if meta.NoText || runner.IsPrivateDataDir(meta.DataDir) {
		// a run made from data-private must be committable: nothing in it may carry item text or checklist rules
		if err := runner.StripChecklistsFile(filepath.Join(dir, "results.json")); err != nil {
			return r, err
		}
		for _, d := range r.Datasets {
			items := filepath.Join(runner.DataPath(root, meta.DataDir, d.Name), "items.jsonl")
			if err := runner.VerifyNoText(filepath.Join(dir, d.Name), items); err != nil {
				return r, err
			}
			if err := runner.ScanFiles([]string{filepath.Join(dir, "results.json"), filepath.Join(dir, "REPORT.md"), filepath.Join(dir, "run.json")}, items); err != nil {
				return r, err
			}
		}
	}
	return r, nil
}

func usStr(x *float64) string {
	if x == nil {
		return "–"
	}
	return fmt.Sprintf("%.0f", *x)
}

// Winners returns the systems in a mode that graduate with accuracy within 1 pt of the lead alone, best first:
// by 2nd-half coverage, ties to the lower mean µs/item.
func Winners(d DatasetResult, mode string) []SystemResult {
	var out []SystemResult
	for _, s := range d.Systems {
		m := s.Metrics
		if s.Mode != mode || m == nil || !m.Graduated || m.AccSystem < m.AccLead-0.01-1e-9 {
			continue
		}
		out = append(out, s)
	}
	us := func(s SystemResult) float64 {
		if s.Metrics.UnderstudentUSMean == nil {
			return math.Inf(1)
		}
		return *s.Metrics.UnderstudentUSMean
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i].Metrics.Coverage2ndHalf, out[j].Metrics.Coverage2ndHalf
		if a != b {
			return a > b
		}
		return us(out[i]) < us(out[j])
	})
	return out
}

func perRespMarkdown(b *strings.Builder, r *Results) {
	found := false
	for _, d := range r.Datasets {
		found = found || d.PerResponse
	}
	if !found {
		return
	}
	w := func(f string, a ...any) { fmt.Fprintf(b, f, a...) }
	w("\n## Per-response datasets: item-level view\n\n")
	w("Each parent item was split into N yes/no questions (ids `<item>#<k>`). **Item pick** = argmax over the item's N final " +
		"per-response probabilities of yes (what callers got); **lead alone** uses the lead's per-response p. " +
		"**Fully local** = items whose N sub-questions were all answered by the understudent: only those save the lead call for the " +
		"whole item (the per-response replay counts calls per sub-question, see `calls saved` in the table above). " +
		"The understudent-only pick on those items is the system pick. Per-response columns are over all sub-items. " +
		"`CV best` is the understudent's best cross-validated agreement over its refits (`majority` = the majority share of the sample).\n\n")
	w("| dataset | system | items | N | item acc lead | item acc system | Δ item acc | fully local | acc lead (fully local) | acc understudent (fully local) | per-resp coverage | per-resp acc lead | per-resp acc system | Δ per-resp acc | CV best | majority |\n")
	w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, d := range r.Datasets {
		if !d.PerResponse {
			continue
		}
		for _, s := range d.Systems {
			if s.Metrics == nil || s.Items == nil {
				w("| %s | %s | | | | | | | | | | | | | | missing: %s |\n", d.Name, s.System, s.Missing)
				continue
			}
			im, m := s.Items, s.Metrics
			cv, maj := "–", "–"
			if s.CV != nil {
				cv, maj = pctp(s.CV.BestAgreement), pctp(s.CV.MajorityShare)
			}
			w("| %s | %s | %d | %d | %s | %s | %+.1f pt | %s (%d) | %s | %s | %s | %s | %s | %+.1f pt | %s | %s |\n",
				d.Name, s.System, im.Items, im.Responses, pct(im.AccLead), pct(im.AccSystem), 100*(im.AccSystem-im.AccLead),
				pct(im.FullyLocalShare), im.FullyLocalN, pctp(im.AccLeadLocal), pctp(im.AccUnderstudentLocal),
				pct(m.Coverage), pct(m.AccLead), pct(m.AccSystem), 100*(m.AccSystem-m.AccLead), cv, maj)
		}
	}
}
