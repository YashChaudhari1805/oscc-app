package main

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

// Audit writes append-only JSONL. It never records images, filenames, e-mail addresses or
// raw IPs: only pseudonymous ids and coarse outcomes.
type Audit struct {
	mu sync.Mutex
	f  *os.File
}

func NewAudit(path string) (*Audit, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	return &Audit{f: f}, nil
}

func (a *Audit) Log(event, user, ip string, kv map[string]any) {
	if a == nil {
		return
	}
	rec := map[string]any{"ts": time.Now().UTC().Format(time.RFC3339), "event": event}
	if user != "" {
		rec["user"] = user
	}
	if ip != "" {
		rec["ip"] = ip
	}
	for k, v := range kv {
		rec[k] = v
	}
	b, _ := json.Marshal(rec)
	a.mu.Lock()
	a.f.Write(append(b, '\n'))
	a.mu.Unlock()
}
