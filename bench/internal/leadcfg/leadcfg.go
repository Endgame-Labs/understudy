// Package leadcfg reads the lead config file shared with the understudy binary: named lead presets and, for the
// bench, endpoint overrides. The file is $UNDERSTUDY_LEADS, else ~/.config/understudy/leads.toml (see
// examples/leads.example.toml). No endpoint is built in: the bench defaults are public vendor URLs.
//
// Only the small TOML subset the file uses is read: `[table]` and `[leads.NAME]` headers, and `key = value` lines
// with string, number or boolean values and `#` comments.
package leadcfg

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Table is one TOML table's scalar values, as strings.
type Table map[string]string

// Config is a parsed lead config file.
type Config struct {
	Path   string
	Tables map[string]Table // "leads.rune", "bench", "rules"
}

// Path is the file the environment names (it may not exist).
func Path() string {
	if p := os.Getenv("UNDERSTUDY_LEADS"); p != "" {
		return p
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "understudy", "leads.toml")
}

// Load reads $UNDERSTUDY_LEADS (which must exist) or the default file (which may not: then an empty config).
func Load() (*Config, error) {
	p := Path()
	f, err := os.Open(p)
	if err != nil {
		if os.IsNotExist(err) && os.Getenv("UNDERSTUDY_LEADS") == "" {
			return &Config{Tables: map[string]Table{}}, nil
		}
		return nil, fmt.Errorf("lead config: %w", err)
	}
	defer f.Close()
	c, err := Parse(bufio.NewScanner(f))
	if err != nil {
		return nil, fmt.Errorf("lead config %s: %w", p, err)
	}
	c.Path = p
	return c, nil
}

// ParseString parses TOML text.
func ParseString(s string) (*Config, error) {
	return Parse(bufio.NewScanner(strings.NewReader(s)))
}

// Parse reads the TOML subset from a scanner.
func Parse(sc *bufio.Scanner) (*Config, error) {
	c := &Config{Tables: map[string]Table{}}
	cur := Table{}
	n := 0
	for sc.Scan() {
		n++
		line := strings.TrimSpace(stripComment(sc.Text()))
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") {
			if !strings.HasSuffix(line, "]") || strings.HasPrefix(line, "[[") {
				return nil, fmt.Errorf("line %d: unsupported table header %q", n, line)
			}
			name := strings.TrimSpace(line[1 : len(line)-1])
			cur = c.Tables[name]
			if cur == nil {
				cur = Table{}
				c.Tables[name] = cur
			}
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("line %d: want key = value, got %q", n, line)
		}
		val, err := scalar(strings.TrimSpace(v))
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
		cur[strings.TrimSpace(k)] = val
	}
	return c, sc.Err()
}

// stripComment removes a trailing `# ...` that is outside a string.
func stripComment(s string) string {
	in, esc := false, false
	for i := 0; i < len(s); i++ {
		switch {
		case esc:
			esc = false
		case in && s[i] == '\\':
			esc = true
		case s[i] == '"':
			in = !in
		case s[i] == '#' && !in:
			return s[:i]
		}
	}
	return s
}

func scalar(v string) (string, error) {
	switch {
	case strings.HasPrefix(v, `"`):
		s, err := strconv.Unquote(v)
		if err != nil {
			return "", fmt.Errorf("bad string %s", v)
		}
		return s, nil
	case strings.HasPrefix(v, `'`) && strings.HasSuffix(v, `'`) && len(v) >= 2:
		return v[1 : len(v)-1], nil
	case v == "":
		return "", fmt.Errorf("missing value")
	case strings.HasPrefix(v, "[") || strings.HasPrefix(v, "{"):
		return "", fmt.Errorf("arrays and inline tables are not supported here: %s", v)
	}
	return v, nil
}

// ErrUnknownLead is wrapped by the error Lead returns for a name with no preset.
var ErrUnknownLead = errors.New("unknown lead")

// Lead is a `[leads.NAME]` preset.
type Lead struct {
	Name, Kind, URL, Model, KeyEnv string
}

// Lead looks a preset up; the error says where presets come from.
func (c *Config) Lead(name string) (Lead, error) {
	t, ok := c.Tables["leads."+name]
	if !ok {
		var known []string
		for k := range c.Tables {
			if n, ok := strings.CutPrefix(k, "leads."); ok {
				known = append(known, n)
			}
		}
		sort.Strings(known)
		switch {
		case c.Path == "":
			return Lead{}, fmt.Errorf("%w %q: no lead config file found (set $UNDERSTUDY_LEADS or create %s; see examples/leads.example.toml), or pass --url and --model", ErrUnknownLead, name, Path())
		case len(known) == 0:
			return Lead{}, fmt.Errorf("%w %q: %s defines no leads; add a [leads.%s] table or pass --url and --model", ErrUnknownLead, name, c.Path, name)
		}
		return Lead{}, fmt.Errorf("%w %q: %s defines %s; add a [leads.%s] table or pass --url and --model", ErrUnknownLead, name, c.Path, strings.Join(known, ", "), name)
	}
	kind := t["kind"]
	if kind == "" {
		kind = "systemone"
	}
	return Lead{Name: name, Kind: kind, URL: t["url"], Model: t["model"], KeyEnv: t["key_env"]}, nil
}

// Setting resolves one endpoint setting: the environment variable, else the config's [bench] key, else def.
func Setting(env, benchKey, def string) string {
	if v := os.Getenv(env); v != "" {
		return v
	}
	if c, err := Load(); err == nil {
		if v := c.Tables["bench"][benchKey]; v != "" {
			return v
		}
	}
	return def
}

// Bearer is the `Authorization` header value for the key in environment variable env, or "" when it is unset.
func Bearer(env string) string {
	if env == "" {
		return ""
	}
	if k := strings.TrimSpace(os.Getenv(env)); k != "" && k != "implicit" {
		return "Bearer " + k
	}
	return ""
}
