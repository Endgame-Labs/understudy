// Package cmd holds understudy-bench's Cobra commands.
package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/dorkitude/understudy/bench/internal/ds"
	"github.com/dorkitude/understudy/bench/internal/llm"
	"github.com/dorkitude/understudy/bench/internal/report"
	"github.com/dorkitude/understudy/bench/internal/runner"
	"github.com/dorkitude/understudy/bench/internal/synth"
	"github.com/dorkitude/understudy/bench/internal/teach"
)

var rootDir string

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, time.Now().Format("15:04:05")+" "+format+"\n", args...)
}

func root() (string, error) {
	if rootDir != "" {
		return filepath.Abs(rootDir)
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return ds.FindRoot(wd)
}

func signalCtx() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
}

// Root returns the understudy-bench command tree.
func Root() *cobra.Command {
	c := &cobra.Command{Use: "understudy-bench", Short: "Benchmark understudy and hunch as reflex layers in front of a decision-model lead",
		SilenceUsage: true}
	c.PersistentFlags().StringVar(&rootDir, "root", "", "understudy repository root (default: the directory above that holds SPEC.md)")
	c.AddCommand(synthCmd(), teachCmd(), runCmd(), reportCmd(), listCmd())
	return c
}

func listCmd() *cobra.Command {
	return &cobra.Command{Use: "datasets", Short: "List the datasets synth can build", RunE: func(*cobra.Command, []string) error {
		for _, d := range synth.Registry() {
			fmt.Printf("%-22s %-18s %-7s %s\n", d.Name, d.Category, d.Type, strings.Join(d.Options, ","))
		}
		return nil
	}}
}

func synthCmd() *cobra.Command {
	var names []string
	var n, conc int
	var seed uint64
	var temp float64
	var url, model string
	c := &cobra.Command{Use: "synth", Short: "Generate datasets with GLM 5.3 Flash (cached in data/<ds>/synth_raw.jsonl, resumable)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			r, err := root()
			if err != nil {
				return err
			}
			sets, err := synth.Lookup(names)
			if err != nil {
				return err
			}
			ctx, cancel := signalCtx()
			defer cancel()
			cl := llm.New(conc)
			cl.URL, cl.Model, cl.Temperature, cl.Log = url, model, temp, logf
			var wg sync.WaitGroup
			var mu sync.Mutex
			var failed []string
			for _, d := range sets {
				wg.Add(1)
				go func() {
					defer wg.Done()
					st, err := synth.Write(ctx, d, cl, filepath.Join(r, "data", d.Name), n, seed, logf)
					mu.Lock()
					defer mu.Unlock()
					if err != nil {
						failed = append(failed, d.Name+": "+err.Error())
						logf("%s: FAILED: %v", d.Name, err)
						return
					}
					logf("%s: wrote %d items, class counts %v %v; batches %v; drops %v", d.Name, st.N, st.ClassCounts, st.Options, st.Batches, st.Drops)
				}()
			}
			wg.Wait()
			logf("GLM calls: %d made, %d from cache, %d failed, %d truncated (finish_reason=length)", cl.Stats.Calls.Load(),
				cl.Stats.Cached.Load(), cl.Stats.Failed.Load(), cl.Stats.Truncated.Load())
			if len(failed) > 0 {
				sort.Strings(failed)
				return fmt.Errorf("synth failed:\n%s", strings.Join(failed, "\n"))
			}
			return nil
		}}
	c.Flags().StringSliceVar(&names, "datasets", nil, "datasets to build (default: all; see `understudy-bench datasets`)")
	c.Flags().IntVar(&n, "n", 600, "items per dataset")
	c.Flags().IntVar(&conc, "concurrency", 8, "concurrent GLM calls (across datasets)")
	c.Flags().Uint64Var(&seed, "seed", 0, "seed for batch plans and stream order")
	c.Flags().Float64Var(&temp, "temperature", 1.0, "sampling temperature")
	c.Flags().StringVar(&url, "url", llm.EndpointURL(), "chat-completions endpoint (default: Fireworks' public API, or $UNDERSTUDY_LLM_URL / [bench] llm_url in the lead config; key from $FIREWORKS_API_KEY)")
	c.Flags().StringVar(&model, "model", llm.DefaultModel, "generator model")
	return c
}

func teachCmd() *cobra.Command {
	var names []string
	var tname, url, model, dataDir string
	var workers int
	c := &cobra.Command{Use: "teach", Short: "Ask a systemone lead every item's question (data/<ds>/lead.<lead>.jsonl, resumable)",
		RunE: func(cmd *cobra.Command, _ []string) error {
			r, err := root()
			if err != nil {
				return err
			}
			p, err := teach.ResolveWith(tname, url, model)
			if err != nil {
				return err
			}
			if url != "" {
				p.URL = url
			}
			if model != "" {
				p.Model = model
			}
			if cmd.Flags().Changed("workers") {
				p.Workers = workers
			}
			if dataDir == "" {
				dataDir = "data"
			}
			if len(names) == 0 {
				names, err = ds.DatasetDirs(runner.DataPath(r, dataDir, ""))
				if err != nil {
					return err
				}
			}
			ctx, cancel := signalCtx()
			defer cancel()
			t := teach.New(p)
			t.Log = logf
			logf("lead %s: %s model %s, %d workers", p.Name, p.URL, p.Model, p.Workers)
			var errs []string
			for _, name := range names {
				t0 := time.Now()
				done, failed, err := t.Dataset(ctx, runner.DataPath(r, dataDir, name))
				if err != nil {
					errs = append(errs, name+": "+err.Error())
					logf("%s: %v", name, err)
					if ctx.Err() != nil {
						break
					}
					continue
				}
				logf("%s: %d answered, %d failed (%.0fs)", name, done, failed, time.Since(t0).Seconds())
				if failed > 0 {
					errs = append(errs, fmt.Sprintf("%s: %d items failed (re-run to retry)", name, failed))
				}
			}
			logf("%s: %d answered, %d already cached, %d failed, %d retries", p.Name, t.Stats.Done.Load(),
				t.Stats.Skipped.Load(), t.Stats.Failed.Load(), t.Stats.Retries.Load())
			if len(errs) > 0 {
				return fmt.Errorf("%s", strings.Join(errs, "\n"))
			}
			return nil
		}}
	c.Flags().StringSliceVar(&names, "datasets", nil, "datasets (default: every <data-dir>/<ds> with a question.json)")
	c.Flags().StringVar(&dataDir, "data-dir", "data", "datasets directory (relative to the repo root unless absolute), e.g. data-private")
	c.Flags().StringVar(&tname, "lead", "rune", "a lead named in the lead config file (e.g. rune, jev); its answers go to data/<ds>/lead.<name>.jsonl")
	c.Flags().StringVar(&url, "url", "", "override the preset's systemone URL")
	c.Flags().StringVar(&model, "model", "", "override the preset's model")
	c.Flags().IntVar(&workers, "workers", 4, "concurrent requests (Rune 429s at ~8)")
	return c
}

func runCmd() *cobra.Command {
	var c runner.Config
	var mode string
	var noText bool
	var ladder, pools []string
	cc := &cobra.Command{Use: "run", Short: "Build streams and run the replay systems",
		Long: "Builds runs/<run>/<ds>/stream.jsonl from items and lead answers (no truth) and runs each system.\n" +
			"--mode replay (default): cold start, each system replays the whole stream.\n" +
			"--mode split: warm start, train on the first --train items' lead answers, then judge the rest.\n" +
			"--mode both: both.\n" +
			"--train-ladder 200,1000,...: a fixed-test learning curve instead: for each rung N, seed each understudent with the first N\n" +
			"lead-labeled items of the dataset's training pool (--train-pool, default <x>_per -> <x>_big_per) and score the dataset itself\n" +
			"with a frozen judge, so every rung is tested on the same held-out items (see `report`'s LADDER.md).",
		RunE: func(cmd *cobra.Command, _ []string) error {
			r, err := root()
			if err != nil {
				return err
			}
			c.Root = r
			if len(ladder) > 0 {
				if c.Ladder, err = runner.ParseLadder(ladder); err != nil {
					return err
				}
				mode = "ladder"
				c.Pools = map[string]string{}
				for _, p := range pools {
					k, v, ok := strings.Cut(p, "=")
					if !ok || k == "" || v == "" {
						return fmt.Errorf("--train-pool wants <test dataset>=<pool dataset>, got %q", p)
					}
					c.Pools[k] = v
				}
			} else if len(pools) > 0 {
				return fmt.Errorf("--train-pool needs --train-ladder")
			}
			switch mode {
			case "replay", "split", "ladder":
				c.Modes = []string{mode}
			case "both":
				c.Modes = []string{"replay", "split"}
			default:
				return fmt.Errorf("--mode must be replay, split or both")
			}
			for i, s := range c.Systems {
				if c.Systems[i], err = runner.NormalizeSystem(s); err != nil {
					return err
				}
			}
			if c.Lead == "" {
				return fmt.Errorf("--lead names the lead whose recorded answers (data/<ds>/lead.<lead>.jsonl) the run replays")
			}
			if c.DataDir == "" {
				c.DataDir = "data"
			}
			c.NoText = noText || (!cmd.Flags().Changed("no-text-artifacts") && runner.IsPrivateDataDir(c.DataDir))
			if len(c.Datasets) == 0 {
				c.Datasets, err = ds.DatasetDirs(runner.DataPath(r, c.DataDir, ""))
				if err != nil {
					return err
				}
			}
			if c.UnderstudyBin == "" {
				c.UnderstudyBin = filepath.Join(r, "target", "release", "understudy")
			}
			c.Log = logf
			ctx, cancel := signalCtx()
			defer cancel()
			return runner.Run(ctx, c)
		}}
	f := cc.Flags()
	f.StringVar(&c.Run, "run", "", "run name (runs/<run>)")
	cc.MarkFlagRequired("run")
	f.StringSliceVar(&c.Systems, "systems", runner.DefaultSystems, "systems: understudent (= understudent:deep) or understudent:<deep|probe|keywords|lite|neighbors|shape|blend|memo|ladder|auto>, or hunch (optional external baseline, see bench/hunch_runner)")
	f.StringSliceVar(&c.Datasets, "datasets", nil, "datasets (default: all in data/)")
	f.StringVar(&c.DataDir, "data-dir", "data", "datasets directory (relative to the repo root unless absolute); data-private holds benchmark text")
	f.BoolVar(&noText, "no-text-artifacts", false, "after the replay delete stream/train/operate/snapshot files and verify nothing left carries state text (automatic with --data-dir data-private; --no-text-artifacts=false disables)")
	f.StringVar(&c.Lead, "lead", "rune", "whose recorded answers to replay (data/<ds>/lead.<name>.jsonl, e.g. rune or jev); no config needed")
	f.StringVar(&mode, "mode", "replay", "replay (cold start), split (warm start) or both")
	f.IntVar(&c.Train, "train", 200, "split mode: stream items used to seed the understudent")
	f.StringSliceVar(&ladder, "train-ladder", nil, "fixed-test learning curve: train sizes, e.g. 200,1000,2000,5000,10000 (the datasets are the held-out tests)")
	f.StringSliceVar(&pools, "train-pool", nil, "with --train-ladder: <test dataset>=<training pool dataset> (repeatable; default <x>_per=<x>_big_per)")
	f.StringVar(&c.UnderstudyBin, "bin", "", "understudy binary (default <root>/target/release/understudy)")
	f.StringSliceVar(&c.UnderstudyArgs, "understudent-arg", nil, "extra argument for understudy replay/judge (repeatable)")
	f.StringVar(&c.EncodeCache, "encode-cache", "", "understudy's on-disk encode cache directory, shared across systems and runs (identical decisions; scratch dirs are git-ignored)")
	f.StringSliceVar(&c.HunchArgs, "hunch-arg", nil, "extra argument for run_hunch.py (repeatable)")
	f.IntVar(&c.Parallel, "parallel", 1, "datasets run concurrently (each replay is CPU-heavy)")
	return cc
}

func reportCmd() *cobra.Command {
	var run, dataDir string
	c := &cobra.Command{Use: "report", Short: "Compute metrics and write runs/<run>/results.json and REPORT.md",
		RunE: func(cmd *cobra.Command, _ []string) error {
			r, err := root()
			if err != nil {
				return err
			}
			res, err := report.Write(r, run, dataDir)
			if err != nil {
				return err
			}
			for _, d := range res.Datasets {
				for _, s := range d.Systems {
					v := "missing: " + s.Missing
					if s.Metrics != nil {
						v = fmt.Sprintf("%s (cov2 %.0f%%, acc %.1f%% vs lead %.1f%%)", s.Metrics.Verdict,
							100*s.Metrics.Coverage2ndHalf, 100*s.Metrics.AccSystem, 100*s.Metrics.AccLead)
					}
					fmt.Printf("%-22s %-14s %s\n", d.Name, s.System, v)
				}
			}
			if len(res.Datasets) == 0 {
				logf("wrote %s", filepath.Join(r, "runs", run, "LADDER.md"))
				return nil
			}
			logf("wrote %s", filepath.Join(r, "runs", run, "REPORT.md"))
			return nil
		}}
	c.Flags().StringVar(&run, "run", "", "run name")
	c.Flags().StringVar(&dataDir, "data-dir", "", "datasets directory (default: the one recorded in the run, else data)")
	c.MarkFlagRequired("run")
	return c
}
