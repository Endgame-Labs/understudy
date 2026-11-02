package synth

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------- code_lookup (binding_lookup, choice), programmatic

var codeOpts = []string{"A", "B", "C", "D"}

// CodeLookup: which candidate's code equals target_code? Generated without GLM.
func CodeLookup() *Dataset {
	d := &Dataset{Name: "code_lookup", Category: "binding_lookup", Type: "choice", Options: codeOpts,
		Question: choiceQuestion("Which candidate's code exactly equals target_code?", codeOpts,
			[]string{"candidate A's code equals target_code", "candidate B's code equals target_code",
				"candidate C's code equals target_code", "candidate D's code equals target_code"}),
		Notes: "binding_lookup: the answer is which key is bound to a value equal to target_code; distractors are " +
			"near misses (one digit changed, two digits swapped, one letter changed). Fully programmatic, balanced. " +
			"Predicted: pooled embeddings carry no usable signal for exact string binding, so neither understudent " +
			"should graduate; never graduating is the correct outcome."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		q := Quotas(n, 0.25, 0.25, 0.25, 0.25)
		var classes []int
		for c, k := range q {
			for i := 0; i < k; i++ {
				classes = append(classes, c)
			}
		}
		r := b.RNG("code", 0)
		const letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
		var out []Unit
		seen := map[string]bool{}
		for _, c := range classes {
			var target string
			for {
				target = fmt.Sprintf("%c%c-%04d", letters[r.IntN(len(letters))], letters[r.IntN(len(letters))], r.IntN(10000))
				if !seen[target] {
					seen[target] = true
					break
				}
			}
			cands := map[string]bool{target: true}
			var distract []string
			for len(distract) < 3 {
				x := []byte(target)
				switch r.IntN(3) {
				case 0: // change one digit
					p := 3 + r.IntN(4)
					x[p] = byte('0' + (int(x[p]-'0')+1+r.IntN(9))%10)
				case 1: // swap two adjacent different digits
					p := 3 + r.IntN(3)
					if x[p] == x[p+1] {
						continue
					}
					x[p], x[p+1] = x[p+1], x[p]
				case 2: // change one letter
					p := r.IntN(2)
					l := letters[r.IntN(len(letters))]
					if l == x[p] {
						continue
					}
					x[p] = l
				}
				s := string(x)
				if !cands[s] {
					cands[s] = true
					distract = append(distract, s)
				}
			}
			m := map[string]any{}
			di := 0
			for k, o := range codeOpts {
				if k == c {
					m[o] = target
				} else {
					m[o] = distract[di]
					di++
				}
			}
			out = append(out, Unit{Class: c, Members: []Member{{State: map[string]any{"target_code": target, "candidates": m}, Truth: c}}})
		}
		return out, nil
	}
	return d
}

// ---------------------------------------------------------------- capital_city (world_knowledge, noul)

var regions = []string{
	"Western Europe", "Northern Europe (Nordics and Baltics)", "Southern Europe", "Central and Eastern Europe",
	"the Balkans", "the Caucasus and Central Asia", "the Middle East", "North Africa", "West Africa",
	"Central Africa", "East Africa", "Southern Africa", "South Asia", "Southeast Asia", "East Asia",
	"Oceania and the Pacific islands", "Central America", "the Caribbean", "South America", "North America",
}

// countries whose capital is split, contested or commonly confused in ways that make a hard label unfair
var capitalExcluded = map[string]bool{
	"bolivia": true, "south africa": true, "netherlands": true, "the netherlands": true, "côte d'ivoire": true,
	"cote d'ivoire": true, "ivory coast": true, "sri lanka": true, "malaysia": true, "benin": true, "eswatini": true,
	"swaziland": true, "israel": true, "palestine": true, "state of palestine": true, "tanzania": true, "taiwan": true,
	"kosovo": true, "western sahara": true, "montenegro": true, "nauru": true, "switzerland": true, "chile": true,
	"micronesia": true, "federated states of micronesia": true, "kiribati": true, "tuvalu": true, "palau": true,
	"burundi": true, "yemen": true, "libya": true, "myanmar": true,
}

func norm(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

// CapitalCity: is the named city the capital of the named country?
func CapitalCity() *Dataset {
	d := &Dataset{Name: "capital_city", Category: "world_knowledge", Type: "noul", Options: noulOpts,
		Question: noulQuestion("Is the named city the capital of the named country?",
			"the city is the country's capital", "the city is not the country's capital"),
		Notes: "world_knowledge: the label is a fact about the world, not about the text; positives and negatives " +
			"look alike (a city and a country). Negatives: 60% a major non-capital city of the same country, 40% the " +
			"capital of another (usually nearby) country. GLM listed capitals (twice per region) and major " +
			"non-capital cities separately; countries with inconsistent answers or split/contested capitals were " +
			"dropped. Predicted: an embedding readout can memorize the pairs it has seen but cannot generalize to " +
			"new countries, so neither understudent should graduate; never graduating is correct."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		var jobs []Job
		type pair struct{ country, city string }
		for rep := 0; rep < 2; rep++ {
			for i, reg := range regions {
				header := fmt.Sprintf("List every sovereign country (UN member or observer state) in %s with its "+
					"official national capital city. Use common English names.", reg)
				job := batchJob(b, fmt.Sprintf("capitals-r%d", rep), i, header, nil, `{"country": "...", "capital": "..."}`, 4000, nil)
				job.User = header + "\n\nReply as {\"items\": [{\"country\": \"...\", \"capital\": \"...\"}, ...]}."
				job.Planned = -1
				jobs = append(jobs, job)
			}
		}
		for rep := 0; rep < 2; rep++ {
			for i, reg := range regions {
				header := fmt.Sprintf("For each sovereign country in %s, list up to 3 of its largest or best-known "+
					"cities that are NOT its national capital (skip a country that has no such city). Use common "+
					"English names.", reg)
				job := batchJob(b, fmt.Sprintf("noncapitals-r%d", rep), i, header, nil, "", 6000, nil)
				job.User = header + "\n\nReply as {\"items\": [{\"country\": \"...\", \"city\": \"...\"}, ...]}."
				job.Planned = -1
				jobs = append(jobs, job)
			}
		}
		// Each job's units: one per listed pair; Class 0 = capital list, 1 = non-capital list. Region kept in Truth.
		for k := range jobs {
			isCap := k < 2*len(regions)
			region := k % len(regions)
			jobs[k].Parse = func(items []map[string]any) []Unit {
				var out []Unit
				for _, m := range items {
					c := Str(m, "country")
					city := Str(m, "capital")
					if !isCap {
						city = Str(m, "city")
					}
					if c == "" || city == "" || len(c) > 60 || len(city) > 60 {
						b.Drop("list_item_malformed")
						continue
					}
					cls := 0
					if !isCap {
						cls = 1
					}
					out = append(out, Unit{Class: cls, Members: []Member{{State: pair{c, city}, Truth: region}}})
				}
				return out
			}
		}
		res := b.Run(ctx, jobs)
		capital := map[string]string{}  // norm country -> norm capital
		display := map[string]string{}  // norm -> display name
		regionOf := map[string]int{}    // norm country -> region
		conflict := map[string]bool{}   // norm country -> inconsistent
		nonCap := map[string][]string{} // norm country -> norm cities
		for _, us := range res {
			for _, u := range us {
				p := u.Members[0].State.(pair)
				cn, ct := norm(p.country), norm(p.city)
				if _, ok := display[cn]; !ok {
					display[cn] = p.country
				}
				if _, ok := display[ct]; !ok {
					display[ct] = p.city
				}
				if u.Class == 0 {
					if old, ok := capital[cn]; ok && old != ct {
						conflict[cn] = true
					}
					capital[cn] = ct
					regionOf[cn] = u.Members[0].Truth
				} else {
					nonCap[cn] = append(nonCap[cn], ct)
				}
			}
		}
		for cn, cities := range nonCap {
			for _, ct := range cities {
				if capital[cn] == ct {
					conflict[cn] = true // listed as both capital and non-capital
				}
			}
		}
		var countries []string
		for cn := range capital {
			if !conflict[cn] && !capitalExcluded[cn] {
				countries = append(countries, cn)
			}
		}
		sort.Strings(countries)
		b.mu.Lock()
		b.Drops["countries_inconsistent"] = len(conflict)
		b.mu.Unlock()
		r := b.RNG("assemble", 0)
		q := Quotas(n, 0.75, 0.25)
		nPos := min(q[1], len(countries))
		nNeg := n - nPos
		if nPos < q[1] {
			b.Log("capital_city: only %d consistent countries for %d positives", len(countries), q[1])
			nNeg = nPos * 3
		}
		used := map[string]bool{}
		var out []Unit
		mk := func(cn, ct string, truth int) bool {
			key := cn + "|" + ct
			if used[key] {
				return false
			}
			used[key] = true
			out = append(out, Unit{Class: truth, Members: []Member{{State: map[string]any{"city": display[ct], "country": display[cn]}, Truth: truth}}})
			return true
		}
		for _, i := range r.Perm(len(countries))[:nPos] {
			mk(countries[i], capital[countries[i]], 1)
		}
		// negatives: 60% non-capital cities, 40% another country's capital (same region when possible)
		var nonPairs [][2]string
		for _, cn := range countries {
			seen := map[string]bool{}
			for _, ct := range nonCap[cn] {
				if !seen[ct] && ct != capital[cn] {
					seen[ct] = true
					nonPairs = append(nonPairs, [2]string{cn, ct})
				}
			}
		}
		sort.Slice(nonPairs, func(i, j int) bool { return nonPairs[i][0]+nonPairs[i][1] < nonPairs[j][0]+nonPairs[j][1] })
		r.Shuffle(len(nonPairs), func(i, j int) { nonPairs[i], nonPairs[j] = nonPairs[j], nonPairs[i] })
		want := nNeg * 6 / 10
		got := 0
		for _, p := range nonPairs {
			if got >= want {
				break
			}
			if mk(p[0], p[1], 0) {
				got++
			}
		}
		for tries := 0; got < nNeg && tries < 100000; tries++ {
			cn := countries[r.IntN(len(countries))]
			var other string
			if r.IntN(3) > 0 { // same region
				var same []string
				for _, o := range countries {
					if o != cn && regionOf[o] == regionOf[cn] {
						same = append(same, o)
					}
				}
				if len(same) > 0 {
					other = same[r.IntN(len(same))]
				}
			}
			if other == "" {
				other = countries[r.IntN(len(countries))]
			}
			if other == cn || capital[other] == capital[cn] {
				continue
			}
			if mk(cn, capital[other], 0) {
				got++
			}
		}
		return out, nil
	}
	return d
}
