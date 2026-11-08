package main

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"net/mail"
	"strings"
	"sync"
	"time"
)

const (
	codeTTL         = 10 * time.Minute
	maxCodeAttempts = 5
	lockoutFor      = 15 * time.Minute
	sessionAbsolute = 12 * time.Hour
	sessionIdle     = 30 * time.Minute
	maxSessionsEach = 3
)

var errBadCode = errors.New("invalid_or_expired_code")

type pendingCode struct {
	mac      []byte
	expires  time.Time
	attempts int
}

type Session struct {
	Email    string
	CSRF     string
	Created  time.Time
	LastSeen time.Time
}

type Auth struct {
	mu       sync.Mutex
	cfg      *Config
	pending  map[string]*pendingCode
	locked   map[string]time.Time
	sessions map[string]*Session // key: sha256(token) hex; raw tokens are never stored
	now      func() time.Time
}

func NewAuth(cfg *Config) *Auth {
	return &Auth{cfg: cfg, pending: map[string]*pendingCode{}, locked: map[string]time.Time{},
		sessions: map[string]*Session{}, now: time.Now}
}

func normalizeEmail(s string) (string, bool) {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" || len(s) > 254 || strings.ContainsAny(s, " \t\r\n<>,;\"") {
		return "", false
	}
	a, err := mail.ParseAddress(s)
	if err != nil || a.Address != s || strings.Count(s, "@") != 1 {
		return "", false
	}
	return s, true
}

func (a *Auth) allowed(email string) bool {
	if a.cfg.AllowAnyEmail || a.cfg.AllowedEmails[email] {
		return true
	}
	return a.cfg.AllowedDomains[email[strings.LastIndex(email, "@")+1:]]
}

func (a *Auth) mac(email, code string) []byte {
	h := hmac.New(sha256.New, a.cfg.SessionSecret)
	h.Write([]byte("otp|" + email + "|" + code))
	return h.Sum(nil)
}

func (a *Auth) UserID(email string) string { // pseudonymous id for logs/audit
	h := hmac.New(sha256.New, a.cfg.SessionSecret)
	h.Write([]byte("uid|" + email))
	return hex.EncodeToString(h.Sum(nil))[:12]
}

func randToken(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return base64.RawURLEncoding.EncodeToString(b)
}

// IssueCode returns (code, true) only when the address is permitted. Callers must respond
// identically either way so the allow-list cannot be enumerated.
func (a *Auth) IssueCode(email string) (string, bool) {
	n, err := rand.Int(rand.Reader, big.NewInt(1_000_000))
	if err != nil {
		panic(err)
	}
	code := fmt.Sprintf("%06d", n.Int64())
	m := a.mac(email, code) // same work whether or not the address is allowed
	if !a.allowed(email) {
		return "", false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if until, ok := a.locked[email]; ok && a.now().Before(until) {
		return "", false
	}
	a.pending[email] = &pendingCode{mac: m, expires: a.now().Add(codeTTL)}
	return code, true
}

// Verify checks a code in constant time. Wrong guesses are counted per issued code; after
// maxCodeAttempts the code is destroyed and the address is locked out.
func (a *Auth) Verify(email, code string) (token string, s *Session, err error) {
	want := a.mac(email, code)
	a.mu.Lock()
	defer a.mu.Unlock()
	now := a.now()
	if until, ok := a.locked[email]; ok && now.Before(until) {
		return "", nil, errBadCode
	}
	p := a.pending[email]
	if p == nil || now.After(p.expires) {
		delete(a.pending, email)
		subtle.ConstantTimeCompare(want, want) // keep timing similar
		return "", nil, errBadCode
	}
	if subtle.ConstantTimeCompare(p.mac, want) != 1 {
		p.attempts++
		if p.attempts >= maxCodeAttempts {
			delete(a.pending, email)
			a.locked[email] = now.Add(lockoutFor)
		}
		return "", nil, errBadCode
	}
	delete(a.pending, email) // single use
	token = randToken(32)
	s = &Session{Email: email, CSRF: randToken(24), Created: now, LastSeen: now}
	a.sessions[tokenKey(token)] = s
	a.capSessions(email)
	return token, s, nil
}

func tokenKey(token string) string {
	h := sha256.Sum256([]byte(token))
	return hex.EncodeToString(h[:])
}

func (a *Auth) capSessions(email string) { // caller holds mu
	var oldestK string
	var oldest time.Time
	count := 0
	for k, s := range a.sessions {
		if s.Email != email {
			continue
		}
		count++
		if oldestK == "" || s.Created.Before(oldest) {
			oldestK, oldest = k, s.Created
		}
	}
	if count > maxSessionsEach {
		delete(a.sessions, oldestK)
	}
}

func (a *Auth) Lookup(token string) *Session {
	if token == "" || len(token) > 128 {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	k := tokenKey(token)
	s := a.sessions[k]
	if s == nil {
		return nil
	}
	now := a.now()
	if now.Sub(s.Created) > sessionAbsolute || now.Sub(s.LastSeen) > sessionIdle || !a.allowed(s.Email) {
		delete(a.sessions, k)
		return nil
	}
	s.LastSeen = now
	cp := *s
	return &cp
}

func (a *Auth) Logout(token string) {
	a.mu.Lock()
	delete(a.sessions, tokenKey(token))
	a.mu.Unlock()
}

func (a *Auth) Sweep() {
	a.mu.Lock()
	defer a.mu.Unlock()
	now := a.now()
	for k, p := range a.pending {
		if now.After(p.expires) {
			delete(a.pending, k)
		}
	}
	for k, u := range a.locked {
		if now.After(u) {
			delete(a.locked, k)
		}
	}
	for k, s := range a.sessions {
		if now.Sub(s.Created) > sessionAbsolute || now.Sub(s.LastSeen) > sessionIdle {
			delete(a.sessions, k)
		}
	}
}
