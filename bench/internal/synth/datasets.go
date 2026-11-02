package synth

import (
	"context"
	"fmt"
	"math/rand/v2"
	"strings"
)

var noulOpts = []string{"false", "true"}

// ---------------------------------------------------------------- tool_failed (surface_property, noul)

var toolNames = []string{"bash", "read_file", "write_file", "edit_file", "grep", "glob", "web_fetch", "run_tests",
	"python", "git", "package_install", "docker", "sql_query", "http_request", "list_dir"}

var failTwists = []string{
	"a non-zero exit with a terse one-line error", "an exception with a stack trace", "permission denied",
	"a timeout", "command or module not found", "a test run with failing tests", "file or path not found",
	"mostly normal-looking output with the error buried in the middle", "a syntax or parse error",
	"a network error (connection refused / DNS / 5xx)", "the edit target string was not found",
	"a merge conflict or rejected git push", "out of memory or disk full",
}

var okTwists = []string{
	"plain successful output", "success whose output contains the words error or fail in content (e.g. grep hits " +
		"for 'error', a test named test_error_handling passing, a log file being read)",
	"empty output, which means success for this tool (say so with a short marker like '(no output)')",
	"success with harmless warnings", "a long successful build or install", "a successful file write/edit confirmation",
	"successful JSON response", "success with exit code 0 shown explicitly",
}

// ToolFailed: did this coding-agent tool call fail?
func ToolFailed() *Dataset {
	d := &Dataset{Name: "tool_failed", Category: "surface_property", Type: "noul", Options: noulOpts,
		Question: noulQuestion("Did this coding-agent tool call fail?",
			"the tool call failed: it errored, was rejected, timed out or did not do what it was asked",
			"the tool call succeeded, even if its output mentions errors in content it found or printed"),
		Notes: "surface_property: failure is usually visible on the surface of the output (tracebacks, exit codes, " +
			"'not found'), with decoy successes that mention 'error'. Predicted: both understudents graduate; understudent's max " +
			"pooling may help on long outputs where the error is buried."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		return fillSimple(ctx, b, n, 10, []float64{0.70, 0.30}, func(r *rand.Rand, i int, classes []int) Job {
			proj := Pick(r, projects)
			var slots []slot
			for _, c := range classes {
				label, twist := "succeeded", Pick(r, okTwists)
				if c == 1 {
					label, twist = "failed", Pick(r, failTwists)
				}
				slots = append(slots, slot{Class: c, Truth: c, Spec: fmt.Sprintf("label=%s; tool=%s; output length=%s; case: %s",
					label, Pick(r, toolNames), Pick(r, lengths), twist)})
			}
			header := fmt.Sprintf("Write tool calls made by an AI coding agent working on %s, each with the raw output the "+
				"tool returned. \"call\" is the tool invocation as the agent wrote it (e.g. a shell command, or a function "+
				"call with arguments); \"output\" is exactly what the tool returned. Do not explain or summarize; the "+
				"output must not state the label in words like 'the call failed' unless a real tool would print that.", proj)
			return batchJob(b, "batch", i, header, slots,
				`{"i": 0, "label": "failed|succeeded", "tool": "...", "call": "...", "output": "..."}`, 12000,
				func(s slot, m map[string]any) (any, string) {
					want := map[int]string{0: "succeeded", 1: "failed"}[s.Class]
					call := Str(m, "call")
					out, ok := StrRaw(m, "output")
					tool := Str(m, "tool")
					switch {
					case !echo(m, want):
						return nil, "label_echo_mismatch"
					case call == "" || tool == "" || !ok:
						return nil, "missing_field"
					case !textOK(out, 1, 6000) || len(call) > 1000:
						return nil, "length_out_of_bounds"
					}
					return map[string]any{"tool": tool, "call": call, "output": out}, ""
				})
		})
	}
	return d
}

// ---------------------------------------------------------------- output_kind (topic_kind, choice)

var outputKinds = []string{"test_results", "file_contents", "error_message", "search_results", "command_output"}

// OutputKind: what kind of output is this?
func OutputKind() *Dataset {
	desc := []string{
		"the report of a test run (test names, pass/fail counts), whether the tests passed or failed",
		"the contents of a file as read from disk (source code, config, docs, data)",
		"an error the tool or command itself raised (stack trace, command not found, permission denied), not a test report",
		"hits from a search (grep/ripgrep/find/code or web search) listing matches and their locations",
		"any other output of a shell command or program (build logs, git status, ls, installs, scripts)",
	}
	d := &Dataset{Name: "output_kind", Category: "topic_kind", Type: "choice", Options: outputKinds,
		Question: choiceQuestion("What kind of output did this coding-agent tool call return?", outputKinds, desc),
		Notes: "topic_kind: the kind of output is a global, topical property of the text. Predicted: both understudents " +
			"graduate quickly (hunch's own pi-stream replay graduated output_kind at item 149)."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		shares := []float64{0.2, 0.2, 0.2, 0.2, 0.2}
		return fillSimple(ctx, b, n, 10, shares, func(r *rand.Rand, i int, classes []int) Job {
			proj := Pick(r, projects)
			var slots []slot
			for _, c := range classes {
				hint := ""
				if r.IntN(4) == 0 {
					hint = "; make it a little ambiguous at first glance but still clearly this kind on reading"
				}
				slots = append(slots, slot{Class: c, Truth: c, Spec: fmt.Sprintf("kind=%s; output length=%s%s",
					outputKinds[c], Pick(r, lengths), hint)})
			}
			header := fmt.Sprintf("Write tool calls made by an AI coding agent working on %s, with the raw output each "+
				"returned. Kinds: %s. \"call\" is the invocation (often a bash command; vary tools), \"output\" is "+
				"exactly what it returned, with no commentary.", proj, strings.Join(func() []string {
				var s []string
				for i, k := range outputKinds {
					s = append(s, k+" = "+desc[i])
				}
				return s
			}(), "; "))
			return batchJob(b, "batch", i, header, slots, `{"i": 0, "label": "<kind>", "call": "...", "output": "..."}`, 12000,
				func(s slot, m map[string]any) (any, string) {
					call := Str(m, "call")
					out, ok := StrRaw(m, "output")
					switch {
					case !echo(m, outputKinds[s.Class]):
						return nil, "label_echo_mismatch"
					case call == "" || !ok || strings.TrimSpace(out) == "":
						return nil, "missing_field"
					case !textOK(out, 5, 6000):
						return nil, "length_out_of_bounds"
					}
					return map[string]any{"call": call, "output": out}, ""
				})
		})
	}
	return d
}

// ---------------------------------------------------------------- customer_frustration (graded_tone, score)

var frustrationLevels = []string{"calm", "mildly annoyed", "frustrated", "furious"}

// CustomerFrustration: how frustrated is the customer?
func CustomerFrustration() *Dataset {
	levels := []string{
		"calm: neutral or positive; asks or reports without negative emotion",
		"mildly annoyed: some irritation or impatience, still polite and measured",
		"frustrated: clearly upset (repeated problems, complaints, demands) but not hostile",
		"furious: enraged or hostile (shouting, insults, threats to cancel, sue or go public)",
	}
	d := &Dataset{Name: "customer_frustration", Category: "graded_tone", Type: "score", Options: []string{"0", "1", "2", "3"},
		Question: scoreQuestion("How frustrated is the customer who wrote this support message?", levels),
		Notes: "graded_tone: an ordinal tone judgment over the whole message. Predicted: both understudents learn the " +
			"extremes; adjacent levels are hard, so graduation needs the confident subset; understudent's pooled features " +
			"may help modestly. Partial takeover is the likely outcome."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		return fillSimple(ctx, b, n, 10, []float64{0.25, 0.25, 0.25, 0.25}, func(r *rand.Rand, i int, classes []int) Job {
			ind := Pick(r, industries)
			var slots []slot
			for _, c := range classes {
				twist := ""
				switch r.IntN(6) {
				case 0:
					twist = "; the emotion is conveyed by content rather than by obvious emotional words"
				case 1:
					twist = "; mention a problem that sounds serious, but keep the stated emotion at this level"
				}
				slots = append(slots, slot{Class: c, Truth: c, Spec: fmt.Sprintf("level=%s; channel=%s; writer style=%s; length=%s%s",
					frustrationLevels[c], Pick(r, channels), Pick(r, writerStyles), Pick(r, []string{"one sentence", "2-4 sentences", "a paragraph", "two paragraphs"}), twist)})
			}
			header := fmt.Sprintf("Write messages that customers of a %s sent to customer support. Each message's emotional "+
				"level must match its assigned level exactly (calm < mildly annoyed < frustrated < furious). Do not "+
				"name the level. Vary the problems (billing, delays, defects, accounts, outages, staff behaviour...).", ind)
			return batchJob(b, "batch", i, header, slots, `{"i": 0, "label": "<level>", "channel": "...", "message": "..."}`, 8000,
				func(s slot, m map[string]any) (any, string) {
					msg := Str(m, "message")
					ch := Str(m, "channel")
					switch {
					case !echo(m, frustrationLevels[s.Class]):
						return nil, "label_echo_mismatch"
					case msg == "" || ch == "":
						return nil, "missing_field"
					case !textOK(msg, 10, 3000):
						return nil, "length_out_of_bounds"
					}
					return map[string]any{"channel": ch, "message": msg}, ""
				})
		})
	}
	return d
}

// ---------------------------------------------------------------- politer_reply (pair_compare, choice a/b)

// PoliterReply: which of two replies is more polite?
func PoliterReply() *Dataset {
	d := &Dataset{Name: "politer_reply", Category: "pair_compare", Type: "choice", Options: []string{"a", "b"},
		Question: choiceQuestion("Two support agents replied to the same customer ticket. Which reply is more polite?",
			[]string{"a", "b"}, []string{"reply a is more polite", "reply b is more polite"}),
		Pair: []string{"a", "b"},
		Notes: "pair_compare: the answer depends on comparing two texts, not on either one alone. Predicted: understudent's " +
			"pair features ([x, y, x-y, x*y]) can learn 'politeness(a) - politeness(b)'; hunch embeds the whole state " +
			"as one mean vector, which mixes both replies, so it should not graduate (correctly abstaining)."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		return fillSimple(ctx, b, n, 10, []float64{0.5, 0.5}, func(r *rand.Rand, i int, classes []int) Job {
			ind := Pick(r, industries)
			var slots []slot
			for _, c := range classes {
				rude := Pick(r, []string{"curt and dismissive", "condescending", "blunt and impatient", "sarcastic",
					"cold and bureaucratic", "openly irritated", "blaming the customer"})
				lenNote := Pick(r, []string{"the polite reply is longer", "the polite reply is shorter",
					"both replies are about the same length"})
				help := Pick(r, []string{"both give the same help", "the less polite reply is actually more helpful",
					"the polite reply is more helpful"})
				slots = append(slots, slot{Class: c, Truth: c, Spec: fmt.Sprintf("less polite reply is %s; %s; %s", rude, lenNote, help)})
			}
			header := fmt.Sprintf("For a %s, write a short customer ticket and two replies to it from support agents, "+
				"with a clear politeness gap between them (one courteous and respectful, one less polite as specified). "+
				"Vary the agents' voices and the problems.", ind)
			return batchJob(b, "batch", i, header, slots, `{"i": 0, "ticket": "...", "polite": "...", "less_polite": "..."}`, 10000,
				func(s slot, m map[string]any) (any, string) {
					p, q := Str(m, "polite"), Str(m, "less_polite")
					switch {
					case p == "" || q == "":
						return nil, "missing_field"
					case p == q:
						return nil, "identical_replies"
					case !textOK(p, 10, 2500) || !textOK(q, 5, 2500):
						return nil, "length_out_of_bounds"
					}
					if s.Class == 0 { // "a" is the more polite
						return map[string]any{"a": p, "b": q}, ""
					}
					return map[string]any{"a": q, "b": p}, ""
				})
		})
	}
	return d
}

// ---------------------------------------------------------------- passage_answers (relational_match, noul)

var negKinds = []string{
	"on the same topic and mentions the query's key entities, but never states the asked fact",
	"answers a closely related but different question (e.g. a different year, person or quantity)",
	"discusses the topic in general and is plausibly relevant, but the specific answer is missing",
	"says the fact is unknown, disputed or not recorded, without giving it",
}

// PassageAnswers: does the passage answer the query?
func PassageAnswers() *Dataset {
	d := &Dataset{Name: "passage_answers", Category: "relational_match", Type: "noul", Options: noulOpts,
		Question: noulQuestion("Does the passage answer the query?",
			"the passage contains the answer to the query",
			"the passage does not answer the query, even if it is on the same topic"),
		Pair: []string{"query", "passage"},
		Notes: "relational_match: the label depends on the relation between query and passage, and negatives are hard " +
			"(same topic, same entities). Predicted: hunch's single mean vector sees topic overlap in both classes and " +
			"should not graduate; understudent's pair features might partly help but answer-containment is a fine-grained " +
			"relation, so abstaining is the expected correct outcome."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		return fillSimple(ctx, b, n, 10, []float64{0.70, 0.30}, func(r *rand.Rand, i int, classes []int) Job {
			tps := PickN(r, topics, 3)
			var slots []slot
			for _, c := range classes {
				spec := "label=answers: the passage states the answer (directly or in a sentence that a careful reader finds)"
				if c == 0 {
					spec = "label=does_not_answer: the passage is " + Pick(r, negKinds)
				}
				spec += fmt.Sprintf("; topic=%s; query style=%s; passage length=%s", Pick(r, tps),
					Pick(r, []string{"a question", "a keyword search query", "an imperative ('find ...')"}),
					Pick(r, []string{"2-3 sentences", "4-6 sentences", "a long paragraph"}))
				slots = append(slots, slot{Class: c, Truth: c, Spec: spec})
			}
			header := "Write search queries, each with a retrieved passage (an encyclopedia- or article-style excerpt). " +
				"Facts may be invented but must be internally consistent. For does_not_answer items the passage must " +
				"be a hard negative: same topic and vocabulary as the query, but without the answer."
			return batchJob(b, "batch", i, header, slots, `{"i": 0, "label": "answers|does_not_answer", "query": "...", "passage": "..."}`, 10000,
				func(s slot, m map[string]any) (any, string) {
					want := map[int]string{0: "does_not_answer", 1: "answers"}[s.Class]
					q, p := Str(m, "query"), Str(m, "passage")
					switch {
					case !echo(m, want):
						return nil, "label_echo_mismatch"
					case q == "" || p == "":
						return nil, "missing_field"
					case !textOK(q, 3, 400) || !textOK(p, 40, 3000):
						return nil, "length_out_of_bounds"
					}
					return map[string]any{"query": q, "passage": p}, ""
				})
		})
	}
	return d
}

// ---------------------------------------------------------------- retry_exception (conditional_rule, noul)

var deployCases = []struct {
	name  string
	truth int
	spec  string
}{
	{"clean_success", 0, "every step succeeds on the first try; the deploy completes"},
	{"retried_then_succeeded", 0, "one step fails (show the real error), is retried automatically, and the retry succeeds; the deploy completes"},
	{"retried_twice_then_succeeded", 0, "a step fails twice, is retried, and the third attempt succeeds; the deploy completes"},
	{"failed_no_retry", 1, "a step fails and there is no retry; the deploy stops or ends failed"},
	{"retried_still_failed", 1, "a step fails, is retried, and every retry fails too; the deploy ends failed"},
	{"retry_ok_later_fail", 1, "one step fails and is retried successfully, but a later step fails without recovery; the deploy ends failed"},
}

// RetryException: did the deploy fail (retried-then-succeeded steps don't count)?
func RetryException() *Dataset {
	d := &Dataset{Name: "retry_exception", Category: "conditional_rule", Type: "noul", Options: noulOpts,
		Question: noulQuestion("Did the deploy fail? A step that failed but was retried and then succeeded does not count as a failure.",
			"the deploy failed: a step failed and was not recovered by a successful retry",
			"the deploy succeeded: no step failed, or every failed step was retried and then succeeded"),
		Notes: "conditional_rule: most negatives contain real error lines (retried then succeeded), so the surface cue " +
			"'there is an error' is wrong about 60% of the time on negatives. Predicted: the understudents learn " +
			"'contains an error' and fail the minority bar, so the correct outcome is never graduating; graduating " +
			"here would likely be harmful."}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		// 30% fail; among the successes, about 60% contain a retried failure
		shares := []float64{0.28, 0.30, 0.12, 0.12, 0.12, 0.06}
		return b.Fill(ctx, Quotas(n, 0.70, 0.30), 10, func(i int) Job {
			r := b.RNG("batch", i)
			tool := Pick(r, deployTools)
			proj := Pick(r, projects)
			var slots []slot
			for _, c := range Assign(r, 10, shares...) {
				dc := deployCases[c]
				slots = append(slots, slot{Class: dc.truth, Truth: dc.truth, Spec: fmt.Sprintf("case=%s: %s; log length=%s",
					dc.name, dc.spec, Pick(r, []string{"15-25 lines", "25-40 lines", "40-60 lines"})), Extra: map[string]any{"case": dc.name}})
			}
			header := fmt.Sprintf("Write deploy logs from %s deploying %s, as the raw console output. Show step names, "+
				"timestamps, real-looking errors and retry messages (e.g. 'Retrying (attempt 2/3)...'). Do not add a "+
				"summary sentence that judges the whole deploy unless the tool really prints one, and vary whether the "+
				"final status line exists.", tool, proj)
			return batchJob(b, "batch", i, header, slots, `{"i": 0, "case": "<case>", "command": "<the deploy command>", "log": "..."}`, 16000,
				func(s slot, m map[string]any) (any, string) {
					cmd, lg := Str(m, "command"), Str(m, "log")
					switch {
					case Str(m, "case") != s.Extra["case"]:
						return nil, "case_echo_mismatch"
					case cmd == "" || lg == "":
						return nil, "missing_field"
					case !textOK(lg, 200, 8000):
						return nil, "length_out_of_bounds"
					}
					return map[string]any{"call": cmd, "output": lg}, ""
				})
		})
	}
	return d
}

// ---------------------------------------------------------------- incident_severity (compound_judgment, score)

var impactNames = []string{"no user-facing effect", "degraded (slow or partly broken)", "outage (a feature or service unavailable)", "data loss or a security breach"}
var scopeNames = []string{"one customer or internal only", "some customers", "most or all customers"}
var severityLevels = []string{"none", "minor", "major", "critical"}

// Severity applies the incident rule stated in the question.
func Severity(impact, scope int, resolved bool) int {
	if impact == 0 {
		return 0
	}
	s, lvl := impact+scope, 1
	switch {
	case s >= 4:
		lvl = 3
	case s == 3:
		lvl = 2
	}
	if !resolved && lvl < 3 {
		lvl++
	}
	return lvl
}

// IncidentSeverity: severity from impact x scope x resolved.
func IncidentSeverity() *Dataset {
	instr := "Rate the incident's severity by combining its impact, its scope and whether it is resolved. Impact: none " +
		"(no user-facing effect), degraded (slow or partly broken), outage (a feature or service unavailable), or data " +
		"loss/security breach. Scope: one customer or internal only, some customers, or most/all customers. An " +
		"incident with no user-facing effect is 'none' whatever else. Otherwise add impact (degraded 1, outage 2, " +
		"data loss/breach 3) and scope (one/internal 0, some 1, most/all 2): a total of 1-2 is minor, 3 is major, 4-5 " +
		"is critical; an unresolved incident is one level higher (at most critical)."
	levels := []string{"none: no user-facing effect", "minor", "major", "critical"}
	d := &Dataset{Name: "incident_severity", Category: "compound_judgment", Type: "score", Options: []string{"0", "1", "2", "3"},
		Question: scoreQuestion(instr, levels),
		Notes: "compound_judgment: the level is a rule over three facts scattered in the report (impact, scope, " +
			"resolved). Generated with balanced levels, each from a uniformly chosen consistent combination. " +
			"Predicted: a linear readout of pooled features can't compute the rule, so the understudents should not " +
			"graduate (correctly abstaining); 'none' vs the rest may be learnable but fails the minority bar."}
	type combo struct {
		impact, scope int
		resolved      bool
	}
	byLevel := make([][]combo, 4)
	for im := 0; im < 4; im++ {
		for sc := 0; sc < 3; sc++ {
			for _, rs := range []bool{true, false} {
				l := Severity(im, sc, rs)
				byLevel[l] = append(byLevel[l], combo{im, sc, rs})
			}
		}
	}
	d.Build = func(ctx context.Context, b *Builder, n int) ([]Unit, error) {
		return fillSimple(ctx, b, n, 10, []float64{0.25, 0.25, 0.25, 0.25}, func(r *rand.Rand, i int, classes []int) Job {
			sys := Pick(r, logSystems)
			var slots []slot
			for _, c := range classes {
				cb := Pick(r, byLevel[c])
				res := "resolved (fixed, verified)"
				if !cb.resolved {
					res = "still unresolved / ongoing"
				}
				slots = append(slots, slot{Class: c, Truth: c,
					Spec: fmt.Sprintf("impact=%s; scope=%s; status=%s; format=%s", impactNames[cb.impact], scopeNames[cb.scope], res,
						Pick(r, []string{"a terse on-call note", "a status-page update", "a Slack incident-channel summary", "a short postmortem paragraph", "a ticket with fields"})),
					Extra: map[string]any{"impact": cb.impact, "scope": cb.scope, "resolved": cb.resolved}})
			}
			header := fmt.Sprintf("Write incident reports for a company whose stack includes %s. Each report must make its "+
				"impact, scope and resolution status clear to a careful reader (through concrete details: numbers of "+
				"customers, what broke, whether a fix is confirmed), but must NOT name a severity or priority: never use "+
				"the words severity, sev, critical, major, minor, P0-P4 or similar.", sys)
			return batchJob(b, "batch", i, header, slots, `{"i": 0, "report": "..."}`, 10000,
				func(s slot, m map[string]any) (any, string) {
					rep := Str(m, "report")
					switch {
					case rep == "":
						return nil, "missing_field"
					case !textOK(rep, 60, 3000):
						return nil, "length_out_of_bounds"
					case severityWords.MatchString(rep):
						return nil, "names_a_severity"
					}
					return map[string]any{"report": rep}, ""
				})
		})
	}
	return d
}
