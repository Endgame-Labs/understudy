// Package report computes per dataset x system metrics from decisions files and writes results.json and
// REPORT.md.
package report

import (
	"math"

	"github.com/dorkitude/understudy/bench/internal/ds"
)

// Metrics for one system on one dataset (over the decisions it made).
type Metrics struct {
	N                     int      `json:"n"`
	Graduated             bool     `json:"graduated"`
	GraduatedAtStart      bool     `json:"graduated_at_start"`      // the first decision already had a mode (split: right after training)
	FirstUnderstudentItem *int     `json:"first_understudent_item"` // 1-based position in these decisions
	FirstUnderstudentID   string   `json:"first_understudent_id,omitempty"`
	Coverage              float64  `json:"coverage"`
	Coverage2ndHalf       float64  `json:"coverage_2nd_half"`
	LocalN                int      `json:"local_n"`
	LocalAgreement        *float64 `json:"local_agreement"`
	AuditN                int      `json:"audit_n"`
	AuditAgreement        *float64 `json:"audit_agreement"`
	MajorityBaseline      *float64 `json:"majority_baseline"` // share of local items whose lead answer is the stream's majority class
	Kappa                 *float64 `json:"kappa"`             // understudent vs lead on local items
	LocalMinorityN        int      `json:"local_minority_n"`
	LocalMinorityRecal    *float64 `json:"local_minority_recall"`
	AccLead               float64  `json:"acc_lead"`
	AccSystem             float64  `json:"acc_system"`
	UnderstudentN         int      `json:"understudent_n"`
	AccUnderstudent       *float64 `json:"acc_understudent_after_fit"`
	LeadCalls             int      `json:"lead_calls"`
	CallsSaved            int      `json:"lead_calls_saved"`
	UnderstudentUSMean    *float64 `json:"understudent_us_mean"`
	Demotions             int      `json:"demotions"`
	Verdict               string   `json:"verdict"`
}

func fp(x float64) *float64 { return &x }

// Kappa is Cohen's kappa between two label sequences over k classes; nil when undefined (no items, or chance
// agreement 1).
func Kappa(a, b []int, k int) *float64 {
	n := len(a)
	if n == 0 || n != len(b) {
		return nil
	}
	ca, cb := make([]float64, k), make([]float64, k)
	agree := 0.0
	for i := range a {
		ca[a[i]]++
		cb[b[i]]++
		if a[i] == b[i] {
			agree++
		}
	}
	po := agree / float64(n)
	pe := 0.0
	for c := 0; c < k; c++ {
		pe += ca[c] / float64(n) * cb[c] / float64(n)
	}
	if pe >= 1 {
		return nil
	}
	return fp((po - pe) / (1 - pe))
}

// Majority returns the most common class (lowest index on ties).
func Majority(y []int, k int) int {
	c := make([]int, k)
	for _, v := range y {
		c[v]++
	}
	return ds.Argmax(func() []float64 {
		f := make([]float64, k)
		for i, x := range c {
			f[i] = float64(x)
		}
		return f
	}())
}

// Compute measures one system's decisions. lead and truth are aligned with dec; majority is the lead's
// majority class over the whole stream.
func Compute(dec []ds.Decision, lead, truth []int, k, majority int) Metrics {
	m := Metrics{N: len(dec)}
	if len(dec) == 0 {
		return m
	}
	half := len(dec) / 2
	var localT, localS []int
	var audits, auditAgree, local2, understudentN, understudentRight, minN, minRight, sysRight, tRight, calls int
	var usSum float64
	var usN int
	fitted := false
	prevDemoted := false
	for i, d := range dec {
		if d.Mode != nil {
			m.Graduated = true
			if i == 0 {
				m.GraduatedAtStart = true
			}
		}
		if d.Demoted && !prevDemoted {
			m.Demotions++
		}
		prevDemoted = d.Demoted
		local := d.AnsweredBy == "understudent"
		if local {
			if m.FirstUnderstudentItem == nil {
				p := i + 1
				m.FirstUnderstudentItem, m.FirstUnderstudentID = &p, d.ID
			}
			m.LocalN++
			if i >= half {
				local2++
			}
			localT = append(localT, lead[i])
			localS = append(localS, d.Label)
			if lead[i] != majority {
				minN++
				if d.Label == lead[i] {
					minRight++
				}
			}
			if d.Audited {
				calls++
			}
		} else {
			calls++
		}
		// audits: local answers checked by the lead, and forwarded items the understudent would have answered
		if d.UnderstudentLabel != nil && ((local && d.Audited) || (!local && d.Eligible)) {
			audits++
			if *d.UnderstudentLabel == lead[i] {
				auditAgree++
			}
		}
		if d.Label == truth[i] {
			sysRight++
		}
		if lead[i] == truth[i] {
			tRight++
		}
		if d.UnderstudentLabel != nil {
			fitted = true
		}
		if fitted && d.UnderstudentLabel != nil {
			understudentN++
			if *d.UnderstudentLabel == truth[i] {
				understudentRight++
			}
		}
		if d.UnderstudentUS != nil {
			usSum += *d.UnderstudentUS
			usN++
		}
	}
	n := float64(len(dec))
	m.Coverage = float64(m.LocalN) / n
	if len(dec)-half > 0 {
		m.Coverage2ndHalf = float64(local2) / float64(len(dec)-half)
	}
	if m.LocalN > 0 {
		agree := 0
		maj := 0
		for i := range localT {
			if localT[i] == localS[i] {
				agree++
			}
			if localT[i] == majority {
				maj++
			}
		}
		m.LocalAgreement = fp(float64(agree) / float64(m.LocalN))
		m.MajorityBaseline = fp(float64(maj) / float64(m.LocalN))
		m.Kappa = Kappa(localT, localS, k)
	}
	m.LocalMinorityN = minN
	if minN > 0 {
		m.LocalMinorityRecal = fp(float64(minRight) / float64(minN))
	}
	m.AuditN = audits
	if audits > 0 {
		m.AuditAgreement = fp(float64(auditAgree) / float64(audits))
	}
	m.AccLead = float64(tRight) / n
	m.AccSystem = float64(sysRight) / n
	m.UnderstudentN = understudentN
	if understudentN > 0 {
		m.AccUnderstudent = fp(float64(understudentRight) / float64(understudentN))
	}
	m.LeadCalls = calls
	m.CallsSaved = len(dec) - calls
	if usN > 0 {
		m.UnderstudentUSMean = fp(usSum / float64(usN))
	}
	m.Verdict = Verdict(m)
	return m
}

// Verdict: "takes over" (2nd-half coverage >= 50% and accuracy within 1 pt of the lead alone), "harmful"
// (graduated and accuracy more than 2 pts below the lead), "correctly abstains" (never graduated), else
// "partial".
func Verdict(m Metrics) string {
	drop := m.AccLead - m.AccSystem
	switch {
	case !m.Graduated:
		return "correctly abstains"
	case drop > 0.02+1e-9:
		return "harmful"
	case m.Coverage2ndHalf >= 0.5 && drop <= 0.01+1e-9:
		return "takes over"
	default:
		return "partial"
	}
}

// round keeps JSON tidy.
func round(x float64, d int) float64 {
	p := math.Pow(10, float64(d))
	return math.Round(x*p) / p
}
