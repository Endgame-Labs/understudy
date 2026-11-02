package synth

import (
	"testing"
)

func TestQuotas(t *testing.T) {
	q := Quotas(600, 0.7, 0.3)
	if q[0]+q[1] != 600 || q[1] != 180 {
		t.Fatal(q)
	}
	q = Quotas(21, 0.25, 0.25, 0.25, 0.25)
	if q[0]+q[1]+q[2]+q[3] != 21 {
		t.Fatal(q)
	}
}

func TestSeverityRule(t *testing.T) {
	cases := []struct {
		im, sc int
		res    bool
		want   int
	}{
		{0, 2, false, 0}, {1, 0, true, 1}, {1, 1, true, 1}, {1, 2, true, 2}, {2, 2, true, 3}, {1, 0, false, 2},
		{2, 1, false, 3}, {3, 0, true, 2}, {3, 2, false, 3},
	}
	for _, c := range cases {
		if got := Severity(c.im, c.sc, c.res); got != c.want {
			t.Errorf("Severity(%d,%d,%v) = %d, want %d", c.im, c.sc, c.res, got, c.want)
		}
	}
}

func TestMinimalPair(t *testing.T) {
	a := "Your payment of $42.10 to Acme Cloud was approved on May 3. Reference 88121."
	if !minimalPair(a, "Your payment of $42.10 to Acme Cloud was declined on May 3. Reference 88121.") {
		t.Error("rejected a one-word flip")
	}
	if minimalPair(a, "Hello! We could not charge your card, please update billing in settings soon.") {
		t.Error("accepted unrelated texts")
	}
}

func TestSecretRegex(t *testing.T) {
	yes := []string{
		"2024-01-01 DEBUG env AWS_SECRET_ACCESS_KEY=abcd1234abcd1234abcd1234",
		"remote: https://ghp_ABCDabcd1234ABCDabcd1234@github.com/x/y.git",
		"Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc",
		"db url postgres://app:s3cr3tpass@db:5432/prod",
		"mysql -u root -p hunter2hunter2",
	}
	no := []string{
		"2024-01-01 INFO commit 3f2a9c1d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a39 deployed",
		"request_id=550e8400-e29b-41d4-a716-446655440000 GET /health 200",
		"pulled image sha256:9b2a1c0f8e7d6c5b4a3928170f6e5d4c3b2a1908f7e6d5c4b3a2918070f6e5d4",
	}
	for _, s := range yes {
		if !secretLike.MatchString(s) {
			t.Errorf("missed secret: %s", s)
		}
	}
	for _, s := range no {
		if secretLike.MatchString(s) {
			t.Errorf("flagged non-secret: %s", s)
		}
	}
	if !redacted.MatchString("api_key=******** sent") {
		t.Error("redacted placeholder not recognized")
	}
}

func TestOrderKeepsPairsApartAndDedups(t *testing.T) {
	d := &Dataset{Name: "t", NoAdjacentGroups: true}
	var units []Unit
	for i := 0; i < 50; i++ {
		units = append(units, Unit{Members: []Member{{State: map[string]any{"n": i, "s": 1}, Truth: 1}, {State: map[string]any{"n": i, "s": 0}, Truth: 0}}})
	}
	units = append(units, Unit{Members: []Member{{State: map[string]any{"n": 0, "s": 1}, Truth: 1}}}) // duplicate
	items, err := Order(d, units, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 100 {
		t.Fatalf("%d items (duplicate not dropped?)", len(items))
	}
	for i := 1; i < len(items); i++ {
		a, b := string(items[i-1].State), string(items[i].State)
		if a[:len(a)-7] == b[:len(b)-7] {
			t.Fatalf("pair members adjacent at %d: %s %s", i, a, b)
		}
	}
	again, _ := Order(d, units, 0)
	for i := range items {
		if string(items[i].State) != string(again[i].State) {
			t.Fatal("order not deterministic")
		}
	}
}
