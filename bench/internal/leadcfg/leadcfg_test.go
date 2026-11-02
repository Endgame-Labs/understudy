package leadcfg

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const sample = `
# a comment
[leads.sys]
url = "https://example.com/v1/systemone"  # trailing comment
model = "m#1"
key_env = "SYS_KEY"

[leads.chatty]
kind = "chat"
url = "https://api.example.com/v1"
logprobs = false

[bench]
llm_url = "https://llm.example.com/v1/chat/completions"
`

func TestParseAndLookup(t *testing.T) {
	c, err := ParseString(sample)
	if err != nil {
		t.Fatal(err)
	}
	l, err := c.Lead("sys")
	if err != nil || l.Kind != "systemone" || l.URL != "https://example.com/v1/systemone" || l.Model != "m#1" || l.KeyEnv != "SYS_KEY" {
		t.Fatalf("%+v %v", l, err)
	}
	ch, _ := c.Lead("chatty")
	if ch.Kind != "chat" || c.Tables["leads.chatty"]["logprobs"] != "false" {
		t.Fatalf("%+v", ch)
	}
	c.Path = "/x/leads.toml"
	if _, err := c.Lead("nope"); err == nil || !strings.Contains(err.Error(), "unknown lead \"nope\"") || !strings.Contains(err.Error(), "chatty, sys") {
		t.Fatalf("%v", err)
	}
	empty, _ := ParseString("")
	if _, err := empty.Lead("rune"); err == nil || !strings.Contains(err.Error(), "no lead config file") {
		t.Fatalf("%v", err)
	}
	for _, bad := range []string{"[leads.x\nurl=1", "url", "k = [1]", "k = \"open"} {
		if _, err := ParseString(bad); err == nil {
			t.Fatalf("accepted %q", bad)
		}
	}
}

func TestSettingPrecedenceAndBearer(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "leads.toml")
	if err := os.WriteFile(p, []byte(sample), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("UNDERSTUDY_LEADS", p)
	t.Setenv("UNDERSTUDY_TEST_LLM_URL", "")
	if got := Setting("UNDERSTUDY_TEST_LLM_URL", "llm_url", "default"); got != "https://llm.example.com/v1/chat/completions" {
		t.Fatalf("config: %s", got)
	}
	t.Setenv("UNDERSTUDY_TEST_LLM_URL", "https://env.example.com")
	if got := Setting("UNDERSTUDY_TEST_LLM_URL", "llm_url", "default"); got != "https://env.example.com" {
		t.Fatalf("env: %s", got)
	}
	if got := Setting("UNDERSTUDY_TEST_LLM_URL", "absent", "default"); got != "https://env.example.com" {
		t.Fatalf("%s", got)
	}
	t.Setenv("UNDERSTUDY_TEST_LLM_URL", "")
	if got := Setting("UNDERSTUDY_TEST_LLM_URL", "absent", "default"); got != "default" {
		t.Fatalf("default: %s", got)
	}
	t.Setenv("UNDERSTUDY_TEST_KEY", " k ")
	if Bearer("UNDERSTUDY_TEST_KEY") != "Bearer k" || Bearer("UNDERSTUDY_TEST_UNSET") != "" || Bearer("") != "" {
		t.Fatal("bearer")
	}
	// a missing $UNDERSTUDY_LEADS file is an error; a missing default file is not
	t.Setenv("UNDERSTUDY_LEADS", filepath.Join(dir, "missing.toml"))
	if _, err := Load(); err == nil {
		t.Fatal("missing explicit file accepted")
	}
	t.Setenv("UNDERSTUDY_LEADS", "")
	t.Setenv("XDG_CONFIG_HOME", dir)
	if c, err := Load(); err != nil || len(c.Tables) != 0 {
		t.Fatalf("%v %v", c, err)
	}
}
