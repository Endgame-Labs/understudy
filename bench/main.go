// Command understudy-bench is the understudy benchmark harness: synth, teach, run, report (see ../SPEC.md).
package main

import (
	"os"

	"github.com/dorkitude/understudy/bench/cmd"
)

func main() {
	if err := cmd.Root().Execute(); err != nil {
		os.Exit(1)
	}
}
