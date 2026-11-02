package synth

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var severityWords = regexp.MustCompile(`(?i)\bseverity\b|\bsev[- ]?[0-4]\b|\bsev\b|\bP[0-4]\b|\bcritical\b`)

// secretLike matches credential-looking material. Haystacks and decoys must not match; needles must.
var secretLike = regexp.MustCompile(`(?i)(api[_-]?key|secret|passw(or)?d|passwd|pwd|token|bearer|private[_ ]key|` +
	`authorization|credential|access[_-]?key)\s*["']?\s*[:=]\s*["']?[A-Za-z0-9/+_\-.]{8,}|` +
	`\bAKIA[0-9A-Z]{8,}|\bgh[pousr]_[A-Za-z0-9]{10,}|\bsk_(live|test)_[A-Za-z0-9]{6,}|\bsk-[A-Za-z0-9_\-]{12,}|` +
	`\bxox[abprs]-[A-Za-z0-9\-]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY|[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^@\s/]{3,}@|` +
	`\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|\bAIza[0-9A-Za-z_\-]{20,}|\bglpat-[A-Za-z0-9_\-]{10,}|` +
	`(?i)(-p|--password)\s+\S{6,}`)

// ---------------------------------------------------------------- secret_needle (needle, noul)

var secretKinds = []string{
	"an AWS access key id and secret access key in an env dump", "a GitHub personal access token in a git remote URL or header",
	"a Stripe live secret key in a config print", "a Slack bot token in a webhook client debug line",
	"an OpenAI-style API key (sk-...) in a request debug log", "a database connection string with a password",
	"an Authorization: Bearer header with a JWT", "a PEM private key header line followed by key material on the same line",
	"a password passed on a command line (e.g. mysql -p...)", "a Google API key (AIza...) in a URL query string",
	"a GitLab token (glpat-...) in a CI variable echo", "a SendGrid or Twilio auth token in a settings dump",
	"basic-auth credentials embedded in an https URL", "a service-account private key id and key in JSON",
}

var decoyKinds = []string{
	"a git commit SHA", "a sha256 checksum of a downloaded artifact", "a UUID request or trace id",
	"a docker image digest (sha256:...)", "a base64-encoded non-secret payload (e.g. a small JSON config or thumbnail)",
	"a redacted credential placeholder like api_key=******** or token=[REDACTED]", "an md5 file hash",
	"a public SSH host key fingerprint", "a long random-looking cache key or ETag", "an npm integrity hash (sha512-...)",
	"a Kubernetes resourceVersion or pod hash suffix", "a session-less CSRF nonce in an HTML form log",
}

// SecretNeedle: does this long log contain a credential anywhere?
func SecretNeedle() *Dataset {
	d := &Dataset{Name: "secret_needle", Category: "needle", Type: "noul", Options: noulOpts,
		Question: noulQuestion("Does this log contain a credential or secret (API key, access token, password, private key) anywhere?",
			"a credential or secret value appears somewhere in the log",
			"no credential or secret appears; hashes, ids, UUIDs, checksums, digests and redacted placeholders don't count"),
		Notes: "needle: one secret line inserted at a random position in a benign ~600-900 token log (GLM writes the " +
			"haystacks; GLM-written fake secrets are inserted by us); half of all logs also get a credential-looking " +
			"decoy (hash, UUID, digest, redacted placeholder). Predicted: understudent's max/attention pooling can pick up " +
			"one salient line and should beat hunch's token mean, which dilutes one line in ~800 tokens. Note: both " +
			"understudents clip states to 3,000 chars (head 2/3, tail 1/3), so a needle in the middle of a long log can " +
			"be invisible to them; the lead reads the whole log."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		nPos := Quotas(n, 0.70, 0.30)[1]
		nDecoy := n / 2
		// haystacks: 3 per call
		hay, err := b.Fill(ctx, []int{n}, 3, func(i int) Job {
			r := b.RNG("haystack", i)
			var slots []slot
			for j := 0; j < 3; j++ {
				slots = append(slots, slot{Spec: fmt.Sprintf("system=%s; about %d lines; %s", Pick(r, logSystems), 40+r.IntN(25),
					Pick(r, []string{"mostly INFO with a few WARN", "includes one handled ERROR", "debug-level verbose",
						"structured JSON lines", "plain text with timestamps", "mixed multi-line stack trace in the middle"}))})
			}
			header := "Write realistic, benign log excerpts (each about 2,500-3,500 characters, i.e. 600-900 tokens). " +
				"They must contain NO credentials, secrets, passwords, tokens, API keys, private keys, connection strings " +
				"with passwords, hashes or long random-looking strings of any kind. Use realistic hostnames, paths, " +
				"timings, user ids (short numbers) and messages. Separate lines with newlines."
			return batchJob(b, "haystack", i, header, slots, `{"i": 0, "log": "..."}`, 12000,
				func(s slot, m map[string]any) (any, string) {
					lg := Str(m, "log")
					switch {
					case lg == "":
						return nil, "haystack_missing"
					case !textOK(lg, 1800, 1<<20):
						return nil, "haystack_too_short"
					case strings.Count(lg, "\n") < 12:
						return nil, "haystack_too_few_lines"
					case secretLike.MatchString(lg):
						return nil, "haystack_contains_secret_like"
					}
					return clipLines(lg, 3600), ""
				})
		})
		if err != nil {
			return nil, err
		}
		lineJob := func(kind string, kinds []string, needle bool) func(i int) Job {
			return func(i int) Job {
				r := b.RNG(kind, i)
				var slots []slot
				for j := 0; j < 12; j++ {
					slots = append(slots, slot{Spec: fmt.Sprintf("%s; written as a log line from %s", Pick(r, kinds), Pick(r, logSystems))})
				}
				header := "Write single log lines (one line each, no newline inside), formatted like the surrounding " +
					"system's logs (timestamp, level, component). "
				if needle {
					header += "Each line must leak the specified credential. Credentials must be invented, never real: " +
						"keep each provider's real format and length (e.g. AKIA + 16 uppercase chars), but make the " +
						"random part obviously fabricated on close inspection (e.g. sequential or repeating runs like " +
						"'ABCD1234ABCD', or 'x' runs). Do not write the words fake, example, dummy, test or sample."
				} else {
					header += "Each line contains the specified credential-looking but NON-secret value (a hash, id, " +
						"digest, or a redacted placeholder). It must not contain any real-looking secret, password or token."
				}
				return batchJob(b, kind, i, header, slots, `{"i": 0, "line": "..."}`, 4000,
					func(s slot, m map[string]any) (any, string) {
						ln := Str(m, "line")
						switch {
						case ln == "" || strings.Contains(ln, "\n"):
							return nil, kind + "_missing_or_multiline"
						case !textOK(ln, 20, 600):
							return nil, kind + "_length_out_of_bounds"
						case needle && !secretLike.MatchString(ln):
							return nil, "needle_not_secret_like"
						case !needle && secretLike.MatchString(ln) && !redacted.MatchString(ln):
							return nil, "decoy_looks_like_secret"
						case needle && regexp.MustCompile(`(?i)fake|example|dummy|sample`).MatchString(ln):
							return nil, "needle_says_fake"
						}
						return ln, ""
					})
			}
		}
		needles, err := b.Fill(ctx, []int{nPos}, 12, lineJob("needle", secretKinds, true))
		if err != nil {
			return nil, err
		}
		decoys, err := b.Fill(ctx, []int{nDecoy}, 12, lineJob("decoy", decoyKinds, false))
		if err != nil {
			return nil, err
		}
		r := b.RNG("assemble", 0)
		pos := make([]bool, n)
		for _, i := range r.Perm(n)[:nPos] {
			pos[i] = true
		}
		dec := make([]bool, n)
		for _, i := range r.Perm(n)[:nDecoy] {
			dec[i] = true
		}
		var out []Unit
		ni, di := 0, 0
		for i := 0; i < n; i++ {
			lines := strings.Split(hay[i].Members[0].State.(string), "\n")
			insert := func(ln string) {
				k := r.IntN(len(lines) + 1)
				lines = append(lines[:k], append([]string{ln}, lines[k:]...)...)
			}
			truth := 0
			if dec[i] {
				insert(decoys[di].Members[0].State.(string))
				di++
			}
			if pos[i] {
				insert(needles[ni].Members[0].State.(string))
				ni++
				truth = 1
			}
			path := Pick(r, []string{"/var/log/app/server.log", "logs/ci-build.txt", "/var/log/syslog", "./worker.log",
				"/srv/deploy/out.log", "kubectl logs deploy/api", "journalctl -u app --since today", "docker logs web-1"})
			call := "cat " + path
			if strings.HasPrefix(path, "kubectl") || strings.HasPrefix(path, "journalctl") || strings.HasPrefix(path, "docker") {
				call = path
			}
			out = append(out, Unit{Class: 0, Members: []Member{{State: map[string]any{"call": call,
				"output": strings.Join(lines, "\n")}, Truth: truth}}})
		}
		return out, nil
	}
	return d
}

var redacted = regexp.MustCompile(`(?i)\*{4,}|\[REDACTED\]|<redacted>|xxxx`)

// ---------------------------------------------------------------- payment_flip (minimal_pair, noul)

var flipKinds = []string{
	"status word: one says the payment succeeded/was approved, the other that it was declined/failed",
	"negation: the other inserts 'not' (or 'could not', 'wasn't') into the key sentence",
	"reversal: the other says the same charge was reversed or refunded back, so the payment did not go through",
	"authorization: one says authorized and captured, the other says authorization failed",
	"one detail in a field: e.g. 'Status: Completed' vs 'Status: Failed' with everything else identical",
	"cause: the other adds a short reason it failed (insufficient funds, expired card, bank declined)",
}

// PaymentFlip: did the payment succeed? Near-identical pairs that differ in one detail.
func PaymentFlip() *Dataset {
	d := &Dataset{Name: "payment_flip", Category: "minimal_pair", Type: "noul", Options: noulOpts,
		Question: noulQuestion("Did the payment succeed?",
			"the payment went through: it was completed and not declined, failed or reversed",
			"the payment did not go through: it was declined, failed, not processed or reversed/refunded"),
		Notes: "minimal_pair: 200 pairs of near-identical notifications (one detail flips success to failure: a " +
			"status word, an inserted 'not', a reversal) plus 200 unpaired successes from the same generator (the " +
			"failure twin discarded), so failure is ~33%. Pair members are never adjacent in the stream. Predicted: " +
			"mean pooling barely moves when one word flips, so hunch should struggle on the failure minority; " +
			"understudent's max/attention pooling may catch the flipped token. Neither is expected to fully take over."}
	d.NoAdjacentGroups = true
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		nPairs := n / 3
		nSingle := n - 2*nPairs
		return b.Fill(ctx, []int{nPairs, nSingle}, 5, func(i int) Job {
			r := b.RNG("batch", i)
			ind := Pick(r, industries)
			var slots []slot
			for j := 0; j < 5; j++ {
				c := 0
				if r.IntN(2) == 0 {
					c = 1 // keep only the success member
				}
				slots = append(slots, slot{Class: c, Spec: fmt.Sprintf("flip=%s; format=%s; amount and currency of your choice",
					Pick(r, flipKinds), Pick(r, []string{"email body", "SMS", "push notification", "webhook JSON payload",
						"bank-app activity entry", "receipt text", "merchant dashboard event"}))})
			}
			header := fmt.Sprintf("Write pairs of payment notifications from a %s (and its payment processor). In each "+
				"pair the two texts are identical except for the one detail specified, which makes the first a payment "+
				"that went through and the second one that did not. Keep names, amounts, ids and wording otherwise the "+
				"same; change as few words as possible.", ind)
			job := batchJob(b, "batch", i, header, slots, `{"i": 0, "succeeded": "...", "failed": "..."}`, 8000, nil)
			job.Parse = func(items []map[string]any) []Unit {
				var out []Unit
				for j, m := range ByIndex(items, len(slots)) {
					ok, bad := Str(m, "succeeded"), Str(m, "failed")
					switch {
					case m == nil:
						b.Drop("item_missing_or_malformed")
						continue
					case ok == "" || bad == "":
						b.Drop("missing_field")
						continue
					case ok == bad:
						b.Drop("pair_identical")
						continue
					case !textOK(ok, 15, 2500) || !textOK(bad, 15, 2500):
						b.Drop("length_out_of_bounds")
						continue
					case !minimalPair(ok, bad):
						b.Drop("pair_not_minimal")
						continue
					}
					u := Unit{Class: slots[j].Class, Members: []Member{{State: map[string]any{"notification": ok}, Truth: 1}}}
					if slots[j].Class == 0 {
						u.Members = append(u.Members, Member{State: map[string]any{"notification": bad}, Truth: 0})
					}
					out = append(out, u)
				}
				return out
			}
			return job
		})
	}
	return d
}

// minimalPair accepts two texts whose word multisets differ by at most 15 words on either side and share at least
// half of the longer text's words (short notifications with an added failure reason differ by ~10 words).
func minimalPair(a, b string) bool {
	wa, wb := words(a), words(b)
	ca := map[string]int{}
	for _, w := range wa {
		ca[w]++
	}
	common := 0
	for _, w := range wb {
		if ca[w] > 0 {
			ca[w]--
			common++
		}
	}
	onlyA, onlyB := len(wa)-common, len(wb)-common
	longer := max(len(wa), len(wb))
	return onlyA <= 15 && onlyB <= 15 && float64(common) >= 0.5*float64(longer)
}

func words(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
}

// clipLines cuts a text at the last line boundary within max bytes (GLM often writes logs longer than asked; the
// spec asks for ~600-900 tokens, i.e. about 2,400-3,600 characters).
func clipLines(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := strings.LastIndexByte(s[:max], '\n')
	if cut < max/2 {
		cut = max
	}
	return strings.TrimRight(s[:cut], "\n")
}
