// Package llm calls GLM 5.3 Flash over an OpenAI-compatible chat-completions endpoint (Fireworks' public API by
// default, the key from $FIREWORKS_API_KEY; override the URL with $UNDERSTUDY_LLM_URL or [bench] llm_url in the lead
// config file) and caches every raw request and response in a JSONL file, keyed so that a re-run reuses earlier
// generations.
package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dorkitude/understudy/bench/internal/ds"
	"github.com/dorkitude/understudy/bench/internal/leadcfg"
	"github.com/dorkitude/understudy/bench/internal/netx"
)

// Defaults for the generator.
const (
	DefaultURL   = "https://api.fireworks.ai/inference/v1/chat/completions"
	DefaultModel = "accounts/fireworks/models/glm-5p3-flash"
	// KeyEnv is the environment variable holding the bearer key.
	KeyEnv = "FIREWORKS_API_KEY"
)

// EndpointURL is the chat-completions URL in effect: $UNDERSTUDY_LLM_URL, else [bench] llm_url in the lead config
// file, else the public default.
func EndpointURL() string { return leadcfg.Setting("UNDERSTUDY_LLM_URL", "llm_url", DefaultURL) }

// Request is the chat-completions body.
type Request struct {
	Model           string            `json:"model"`
	Messages        []Message         `json:"messages"`
	Temperature     float64           `json:"temperature"`
	MaxTokens       int               `json:"max_tokens"`
	ReasoningEffort string            `json:"reasoning_effort,omitempty"`
	ResponseFormat  map[string]string `json:"response_format,omitempty"`
}

// Message is one chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Record is one line of synth_raw.jsonl.
type Record struct {
	Key          string          `json:"key"`
	Time         string          `json:"time"`
	Request      Request         `json:"request"`
	Response     json.RawMessage `json:"response,omitempty"`
	Content      string          `json:"content"`
	FinishReason string          `json:"finish_reason,omitempty"`
	Error        string          `json:"error,omitempty"`
	Attempts     int             `json:"attempts"`
	LatencyMS    int64           `json:"latency_ms"`
}

// Stats counts calls.
type Stats struct {
	Calls, Cached, Failed, Truncated atomic.Int64
}

// Client generates with caching. One Client may serve several datasets (each with its own cache file); Sem bounds
// the concurrent calls across all of them.
type Client struct {
	URL, Model  string
	Temperature float64
	Effort      string
	HTTP        *netx.Client
	Sem         chan struct{}
	Stats       Stats
	Log         func(format string, args ...any)
}

// New returns a client with `concurrency` concurrent calls.
func New(concurrency int) *Client {
	h := netx.New(5*time.Minute, 6)
	h.Backoff = 3 * time.Second
	if b := leadcfg.Bearer(KeyEnv); b != "" {
		h.Headers = map[string]string{"Authorization": b}
	}
	return &Client{URL: EndpointURL(), Model: DefaultModel, Temperature: 1.0, Effort: "low", HTTP: h,
		Sem: make(chan struct{}, concurrency), Log: func(string, ...any) {}}
}

// Cache is one dataset's synth_raw.jsonl: the newest successful record per key.
type Cache struct {
	mu   sync.Mutex
	recs map[string]Record
	app  *ds.Appender
}

// OpenCache loads and opens a cache file for appending.
func OpenCache(path string) (*Cache, error) {
	rows, err := ds.ReadJSONL[Record](path)
	if err != nil {
		return nil, err
	}
	c := &Cache{recs: map[string]Record{}}
	for _, r := range rows {
		if r.Error == "" && r.Content != "" {
			c.recs[r.Key] = r
		}
	}
	c.app, err = ds.OpenAppender(path)
	return c, err
}

// Close closes the cache file.
func (c *Cache) Close() error { return c.app.Close() }

// Len is the number of cached successful generations.
func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.recs)
}

// Key builds a cache key from a logical name and the prompt, so editing a prompt template regenerates.
func Key(name string, system, user string, maxTokens int) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%s\x00%d", system, user, maxTokens)))
	return name + "#" + hex.EncodeToString(h[:6])
}

// Complete returns the content for (system, user), from the cache when it has it.
func (c *Client) Complete(ctx context.Context, cache *Cache, name, system, user string, maxTokens int) (string, error) {
	key := Key(name, system, user, maxTokens)
	cache.mu.Lock()
	r, ok := cache.recs[key]
	cache.mu.Unlock()
	if ok {
		c.Stats.Cached.Add(1)
		return r.Content, nil
	}
	req := Request{Model: c.Model, Temperature: c.Temperature, MaxTokens: maxTokens, ReasoningEffort: c.Effort,
		ResponseFormat: map[string]string{"type": "json_object"},
		Messages:       []Message{{Role: "system", Content: system}, {Role: "user", Content: user}}}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	select {
	case c.Sem <- struct{}{}:
	case <-ctx.Done():
		return "", ctx.Err()
	}
	t0 := time.Now()
	resp, attempts, err := c.HTTP.PostJSON(ctx, c.URL, body)
	<-c.Sem
	c.Stats.Calls.Add(1)
	rec := Record{Key: key, Time: t0.UTC().Format(time.RFC3339), Request: req, Attempts: attempts,
		LatencyMS: time.Since(t0).Milliseconds()}
	if err == nil {
		var out struct {
			Choices []struct {
				Message      struct{ Content string } `json:"message"`
				FinishReason string                   `json:"finish_reason"`
			} `json:"choices"`
		}
		if jerr := json.Unmarshal(resp, &out); jerr != nil || len(out.Choices) == 0 {
			err = fmt.Errorf("unparsable response: %v", jerr)
		} else {
			rec.Content = out.Choices[0].Message.Content
			rec.FinishReason = out.Choices[0].FinishReason
			if rec.Content == "" {
				err = errors.New("empty content")
			}
		}
		rec.Response = resp
	}
	if err != nil {
		c.Stats.Failed.Add(1)
		rec.Error = err.Error()
		rec.Content = ""
		c.Log("glm %s failed after %d attempts: %v", name, attempts, err)
	}
	if rec.FinishReason == "length" {
		c.Stats.Truncated.Add(1)
	}
	if aerr := cache.app.Append(rec); aerr != nil {
		return "", aerr
	}
	if err != nil {
		return "", err
	}
	cache.mu.Lock()
	cache.recs[key] = rec
	cache.mu.Unlock()
	return rec.Content, nil
}

// ParseItems extracts the "items" array of a JSON object reply (tolerating a ```json fence).
func ParseItems(content string) ([]map[string]any, error) {
	s := strings.TrimSpace(content)
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	}
	var obj struct {
		Items []json.RawMessage `json:"items"`
	}
	if err := json.Unmarshal([]byte(s), &obj); err != nil {
		return nil, fmt.Errorf("not a JSON object with items: %w", err)
	}
	if obj.Items == nil {
		return nil, errors.New("no items array")
	}
	out := make([]map[string]any, 0, len(obj.Items))
	for _, raw := range obj.Items {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			out = append(out, nil) // a malformed item: the caller drops it
			continue
		}
		out = append(out, m)
	}
	return out, nil
}
