package main

import (
	"sync"
	"time"
)

// Rule is one sliding-window limit: at most Limit events per Window for (Name, Key).
type Rule struct {
	Name, Key string
	Limit     int
	Window    time.Duration
}

type bucket struct {
	ts     []time.Time
	window time.Duration
}

// Limiter is an in-memory sliding-window-log limiter. Allow is all-or-nothing across the
// supplied rules (a request denied by one rule consumes no quota on the others).
// Single-node only; swap for Redis if you run several gateway replicas.
type Limiter struct {
	mu      sync.Mutex
	hits    map[string]*bucket
	now     func() time.Time
	maxKeys int
}

func NewLimiter() *Limiter {
	return &Limiter{hits: map[string]*bucket{}, now: time.Now, maxKeys: 200_000}
}

func (l *Limiter) Allow(rules ...Rule) (bool, time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	var retry time.Duration
	denied := false
	type pend struct {
		k string
		b *bucket
	}
	var ok []pend
	for _, r := range rules {
		k := r.Name + "|" + r.Key
		b := l.hits[k]
		if b == nil {
			if len(l.hits) >= l.maxKeys { // memory-exhaustion guard: fail closed for new keys
				return false, time.Minute
			}
			b = &bucket{window: r.Window}
		}
		cut := now.Add(-r.Window)
		i := 0
		for i < len(b.ts) && !b.ts[i].After(cut) {
			i++
		}
		b.ts = b.ts[i:]
		if len(b.ts) >= r.Limit {
			denied = true
			if d := b.ts[len(b.ts)-r.Limit].Add(r.Window).Sub(now); d > retry {
				retry = d
			}
		}
		ok = append(ok, pend{k, b})
	}
	if denied {
		if retry < time.Second {
			retry = time.Second
		}
		return false, retry
	}
	for _, p := range ok {
		p.b.ts = append(p.b.ts, now)
		l.hits[p.k] = p.b
	}
	return true, 0
}

// Sweep drops buckets whose newest event is older than their window.
func (l *Limiter) Sweep() {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	for k, b := range l.hits {
		if len(b.ts) == 0 || b.ts[len(b.ts)-1].Add(b.window).Before(now) {
			delete(l.hits, k)
		}
	}
}
