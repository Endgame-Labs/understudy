package netx

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestRetryAfter(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if RetryAfter("3", now) != 3*time.Second || RetryAfter("", now) != 0 || RetryAfter("junk", now) != 0 {
		t.Fatal("seconds")
	}
	if d := RetryAfter(now.Add(5*time.Second).Format(http.TimeFormat), now); d != 5*time.Second {
		t.Fatalf("date: %v", d)
	}
}

func TestPostJSONRetries429(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n.Add(1) < 3 {
			w.Header().Set("Retry-After", "0.01")
			w.WriteHeader(429)
			return
		}
		w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()
	c := New(5*time.Second, 4)
	b, attempts, err := c.PostJSON(context.Background(), srv.URL, []byte(`{}`))
	if err != nil || attempts != 3 || string(b) != `{"ok":true}` {
		t.Fatalf("%s %d %v", b, attempts, err)
	}
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(400) }))
	defer srv2.Close()
	if _, attempts, err := c.PostJSON(context.Background(), srv2.URL, nil); err == nil || attempts != 1 {
		t.Fatalf("400 retried: %d %v", attempts, err)
	}
}
