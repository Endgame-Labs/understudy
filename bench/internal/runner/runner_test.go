package runner

import (
	"strings"
	"testing"
)

func TestNormalizeSystem(t *testing.T) {
	cases := map[string]string{"hunch": "hunch", "understudent": "understudent-deep",
		"understudent:probe": "understudent-probe", "understudent-neighbors": "understudent-neighbors"}
	for in, want := range cases {
		if got, err := NormalizeSystem(in); err != nil || got != want {
			t.Errorf("%s: %s %v", in, got, err)
		}
	}
	for _, bad := range []string{"understudent:bert", "jev", ""} {
		if _, err := NormalizeSystem(bad); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
	if SystemName("understudent-neighbors", "split") != "understudent-neighbors-split" {
		t.Error("split name")
	}
}

func TestLastJSONObject(t *testing.T) {
	pretty := "progress 1/2\n{\"a\": 1}\nmore\n{\n  \"items\": 20,\n  \"x\": {\n    \"y\": null\n  }\n}\n"
	if got := string(LastJSONObject([]byte(pretty))); got != `{"items":20,"x":{"y":null}}` {
		t.Fatalf("pretty: %s", got)
	}
	if got := string(LastJSONObject([]byte("log\n{\"system\": \"hunch\", \"n\": 3}\n"))); got != `{"system":"hunch","n":3}` {
		t.Fatalf("one line: %s", got)
	}
	if LastJSONObject([]byte("Traceback\nValueError: x\n")) != nil {
		t.Fatal("found an object in a traceback")
	}
}

func TestUnderstudyExtraAddsEncodeCache(t *testing.T) {
	c := Config{Root: "/r", UnderstudyArgs: []string{"--target", "0.97"}}
	if got := strings.Join(c.UnderstudyExtra(), " "); got != "--target 0.97" {
		t.Fatal(got)
	}
	c.EncodeCache = ".claude/scratch/x"
	if got := strings.Join(c.UnderstudyExtra(), " "); got != "--target 0.97 --encode-cache /r/.claude/scratch/x" {
		t.Fatal(got)
	}
	c.EncodeCache = "/abs"
	if got := c.UnderstudyExtra(); got[len(got)-1] != "/abs" || len(c.UnderstudyArgs) != 2 {
		t.Fatal(got, c.UnderstudyArgs)
	}
}

func TestNewUnderstudentSystemNamesAreAccepted(t *testing.T) {
	for _, n := range []string{"shape", "blend", "memo", "net", "checklist"} {
		for _, spec := range []string{"understudent:" + n, "understudent-" + n} {
			got, err := NormalizeSystem(spec)
			if err != nil || got != "understudent-"+n {
				t.Fatalf("%s: %q %v", spec, got, err)
			}
		}
	}
}

func TestHunchIsAnOptionalBaseline(t *testing.T) {
	for _, s := range DefaultSystems {
		if s == "hunch" {
			t.Fatal("hunch must not be a default system")
		}
	}
	if n, err := NormalizeSystem("hunch"); err != nil || n != "hunch" {
		t.Fatalf("hunch still selectable: %q %v", n, err)
	}
}
