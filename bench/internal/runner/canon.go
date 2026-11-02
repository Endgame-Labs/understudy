package runner

// A port of the Rust side's state identity: src/encoder.rs state_text over src/pyjson.rs (dumps, py_str, float_repr).
// The fit dedupes pool items by this text, so the ladder must count duplicates the same way: states that differ only in
// key order, spacing or number spelling are one state.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

// jv is a parsed JSON value that keeps object key order (serde_json's preserve_order).
type jv struct {
	kind byte // n null, b bool, i int (text in num), f float, s string, a array, o object
	b    bool
	num  string
	f    float64
	s    string
	a    []jv
	keys []string
	vals []jv
}

func parseJV(raw []byte) (jv, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := readJV(dec)
	if err != nil {
		return jv{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return jv{}, fmt.Errorf("trailing data after state")
	}
	return v, nil
}

func readJV(dec *json.Decoder) (jv, error) {
	t, err := dec.Token()
	if err != nil {
		return jv{}, err
	}
	switch x := t.(type) {
	case nil:
		return jv{kind: 'n'}, nil
	case bool:
		return jv{kind: 'b', b: x}, nil
	case string:
		return jv{kind: 's', s: x}, nil
	case json.Number:
		return numJV(string(x))
	case json.Delim:
		if x == '[' {
			v := jv{kind: 'a'}
			for dec.More() {
				e, err := readJV(dec)
				if err != nil {
					return jv{}, err
				}
				v.a = append(v.a, e)
			}
			_, err := dec.Token()
			return v, err
		}
		v := jv{kind: 'o'}
		for dec.More() {
			kt, err := dec.Token()
			if err != nil {
				return jv{}, err
			}
			e, err := readJV(dec)
			if err != nil {
				return jv{}, err
			}
			k := kt.(string)
			if i := indexOf(v.keys, k); i >= 0 { // a repeated key keeps its first position and takes the last value
				v.vals[i] = e
			} else {
				v.keys, v.vals = append(v.keys, k), append(v.vals, e)
			}
		}
		_, err = dec.Token()
		return v, err
	}
	return jv{}, fmt.Errorf("unexpected token %v", t)
}

func indexOf(ks []string, k string) int {
	for i, x := range ks {
		if x == k {
			return i
		}
	}
	return -1
}

// numJV mirrors serde_json: an integer literal that fits i64/u64 is an integer, anything else an f64 ("-0" is -0.0).
func numJV(s string) (jv, error) {
	if !strings.ContainsAny(s, ".eE") && s != "-0" {
		if _, err := strconv.ParseInt(s, 10, 64); err == nil {
			return jv{kind: 'i', num: s}, nil
		}
		if _, err := strconv.ParseUint(s, 10, 64); err == nil {
			return jv{kind: 'i', num: s}, nil
		}
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil && !math.IsInf(f, 0) {
		return jv{}, err
	}
	return jv{kind: 'f', f: f}, nil
}

// CanonicalState is the Rust state_text of a state given as JSON.
func CanonicalState(raw []byte) (string, error) {
	v, err := parseJV(raw)
	if err != nil {
		return "", err
	}
	return stateText(v), nil
}

func (v jv) get(k string) (jv, bool) {
	if i := indexOf(v.keys, k); i >= 0 {
		return v.vals[i], true
	}
	return jv{}, false
}

func stateText(v jv) string {
	if v.kind == 's' {
		return v.s
	}
	if v.kind == 'o' {
		call, ok := v.get("call")
		if !ok {
			call, ok = v.get("tool_call")
		}
		out, hasOut := v.get("output")
		if ok && call.kind != 'n' && hasOut {
			var sb strings.Builder
			sb.WriteString(pyStr(call) + "\n" + pyStr(out))
			var keys []string
			for _, k := range v.keys {
				if k != "call" && k != "tool_call" && k != "output" {
					keys = append(keys, k)
				}
			}
			sort.Strings(keys) // byte order of UTF-8 is code point order
			for _, k := range keys {
				x, _ := v.get(k)
				s := x.s
				if x.kind != 's' {
					s = pyDumps(x)
				}
				sb.WriteString("\n" + k + ": " + s)
			}
			return sb.String()
		}
	}
	return pyDumps(v)
}

func pyDumps(v jv) string {
	var sb strings.Builder
	dump(v, &sb)
	return sb.String()
}

func dump(v jv, sb *strings.Builder) {
	switch v.kind {
	case 'n':
		sb.WriteString("null")
	case 'b':
		sb.WriteString(strconv.FormatBool(v.b))
	case 'i':
		sb.WriteString(v.num)
	case 'f':
		sb.WriteString(floatRepr(v.f))
	case 's':
		quote(v.s, sb)
	case 'a':
		sb.WriteByte('[')
		for i, x := range v.a {
			if i > 0 {
				sb.WriteString(", ")
			}
			dump(x, sb)
		}
		sb.WriteByte(']')
	case 'o':
		idx := make([]int, len(v.keys))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return v.keys[idx[a]] < v.keys[idx[b]] })
		sb.WriteByte('{')
		for n, i := range idx {
			if n > 0 {
				sb.WriteString(", ")
			}
			quote(v.keys[i], sb)
			sb.WriteString(": ")
			dump(v.vals[i], sb)
		}
		sb.WriteByte('}')
	}
}

func quote(s string, sb *strings.Builder) {
	sb.WriteByte('"')
	for _, c := range s {
		switch {
		case c == '"':
			sb.WriteString(`\"`)
		case c == '\\':
			sb.WriteString(`\\`)
		case c == '\n':
			sb.WriteString(`\n`)
		case c == '\r':
			sb.WriteString(`\r`)
		case c == '\t':
			sb.WriteString(`\t`)
		case c == 8:
			sb.WriteString(`\b`)
		case c == 12:
			sb.WriteString(`\f`)
		case c < 0x20:
			fmt.Fprintf(sb, `\u%04x`, c)
		default:
			sb.WriteRune(c)
		}
	}
	sb.WriteByte('"')
}

// floatRepr is Python's repr(float): shortest round-tripping digits, scientific below 1e-4 and from 1e16.
func floatRepr(x float64) string {
	switch {
	case math.IsNaN(x):
		return "NaN"
	case math.IsInf(x, 1):
		return "Infinity"
	case math.IsInf(x, -1):
		return "-Infinity"
	case x == 0:
		if math.Signbit(x) {
			return "-0.0"
		}
		return "0.0"
	}
	e := strconv.FormatFloat(x, 'e', -1, 64)
	mant, expS, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expS)
	sign := ""
	if strings.HasPrefix(mant, "-") {
		sign = "-"
	}
	digits := strings.ReplaceAll(strings.TrimPrefix(mant, "-"), ".", "")
	if exp >= -4 && exp < 16 {
		n := len(digits)
		switch {
		case exp >= n-1:
			return sign + digits + strings.Repeat("0", exp-(n-1)) + ".0"
		case exp >= 0:
			return sign + digits[:exp+1] + "." + digits[exp+1:]
		default:
			return sign + "0." + strings.Repeat("0", -exp-1) + digits
		}
	}
	m := digits
	if len(digits) > 1 {
		m = digits[:1] + "." + digits[1:]
	}
	es := "+"
	if exp < 0 {
		es, exp = "-", -exp
	}
	return fmt.Sprintf("%s%se%s%02d", sign, m, es, exp)
}

// pyStr is Python's str(v) of a json.loads value: a string as is, else its repr.
func pyStr(v jv) string {
	if v.kind == 's' {
		return v.s
	}
	var sb strings.Builder
	repr(v, &sb)
	return sb.String()
}

func repr(v jv, sb *strings.Builder) {
	switch v.kind {
	case 'n':
		sb.WriteString("None")
	case 'b':
		if v.b {
			sb.WriteString("True")
		} else {
			sb.WriteString("False")
		}
	case 'i':
		sb.WriteString(v.num)
	case 'f':
		switch s := floatRepr(v.f); s {
		case "NaN":
			sb.WriteString("nan")
		case "Infinity":
			sb.WriteString("inf")
		case "-Infinity":
			sb.WriteString("-inf")
		default:
			sb.WriteString(s)
		}
	case 's':
		strRepr(v.s, sb)
	case 'a':
		sb.WriteByte('[')
		for i, x := range v.a {
			if i > 0 {
				sb.WriteString(", ")
			}
			repr(x, sb)
		}
		sb.WriteByte(']')
	case 'o':
		sb.WriteByte('{')
		for i, k := range v.keys {
			if i > 0 {
				sb.WriteString(", ")
			}
			strRepr(k, sb)
			sb.WriteString(": ")
			repr(v.vals[i], sb)
		}
		sb.WriteByte('}')
	}
}

func strRepr(s string, sb *strings.Builder) {
	q := '\''
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		q = '"'
	}
	sb.WriteRune(q)
	for _, c := range s {
		switch {
		case c == '\\':
			sb.WriteString(`\\`)
		case c == '\n':
			sb.WriteString(`\n`)
		case c == '\r':
			sb.WriteString(`\r`)
		case c == '\t':
			sb.WriteString(`\t`)
		case c == q:
			sb.WriteRune('\\')
			sb.WriteRune(c)
		case c < 0x20 || (c >= 0x7f && c < 0xa1):
			fmt.Fprintf(sb, `\x%02x`, c)
		default:
			sb.WriteRune(c)
		}
	}
	sb.WriteRune(q)
}
