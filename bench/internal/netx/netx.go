// Package netx posts JSON with retries: 408, 429, 5xx and transport errors are retried with exponential backoff
// (with jitter), honoring Retry-After when the server sends one.
package netx

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Client posts JSON bodies with retries.
type Client struct {
	HTTP        *http.Client
	MaxAttempts int           // first attempt plus retries
	Backoff     time.Duration // base sleep, doubled per attempt
	MaxSleep    time.Duration // cap for one sleep (Retry-After included)
	Headers     map[string]string
	OnRetry     func(attempt int, status int, err error, sleep time.Duration)
}

// New returns a client with the given per-request timeout and attempts.
func New(timeout time.Duration, attempts int) *Client {
	return &Client{HTTP: &http.Client{Timeout: timeout}, MaxAttempts: attempts, Backoff: 2 * time.Second,
		MaxSleep: 2 * time.Minute}
}

// StatusError is a final non-2xx response.
type StatusError struct {
	Status int
	Body   string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.Status, truncate(e.Body, 300))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// Retryable reports whether a status is worth retrying.
func Retryable(status int) bool {
	return status == 408 || status == 409 || status == 425 || status == 429 || status >= 500
}

// RetryAfter parses a Retry-After header (seconds or an HTTP date); 0 when absent or unparsable.
func RetryAfter(h string, now time.Time) time.Duration {
	h = strings.TrimSpace(h)
	if h == "" {
		return 0
	}
	if s, err := strconv.ParseFloat(h, 64); err == nil && s >= 0 {
		return time.Duration(s * float64(time.Second))
	}
	if t, err := http.ParseTime(h); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// PostJSON posts body and returns the 2xx response body, the attempts used and the last error.
func (c *Client) PostJSON(ctx context.Context, url string, body []byte) ([]byte, int, error) {
	var lastErr error
	for attempt := 1; attempt <= c.MaxAttempts; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, attempt, err
		}
		req.Header.Set("Content-Type", "application/json")
		for k, v := range c.Headers {
			req.Header.Set(k, v)
		}
		resp, err := c.HTTP.Do(req)
		var sleep time.Duration
		status := 0
		if err == nil {
			status = resp.StatusCode
			b, rerr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if rerr == nil && status >= 200 && status < 300 {
				return b, attempt, nil
			}
			if rerr != nil {
				lastErr = rerr
			} else {
				lastErr = &StatusError{Status: status, Body: string(b)}
				if !Retryable(status) {
					return nil, attempt, lastErr
				}
			}
			sleep = RetryAfter(resp.Header.Get("Retry-After"), time.Now())
		} else {
			if ctx.Err() != nil {
				return nil, attempt, ctx.Err()
			}
			lastErr = err
		}
		if attempt == c.MaxAttempts {
			break
		}
		if sleep == 0 {
			sleep = c.Backoff << (attempt - 1)
			sleep += time.Duration(rand.Int64N(int64(sleep)/2 + 1))
		}
		if c.MaxSleep > 0 && sleep > c.MaxSleep {
			sleep = c.MaxSleep
		}
		if c.OnRetry != nil {
			c.OnRetry(attempt, status, lastErr, sleep)
		}
		select {
		case <-ctx.Done():
			return nil, attempt, ctx.Err()
		case <-time.After(sleep):
		}
	}
	return nil, c.MaxAttempts, lastErr
}
