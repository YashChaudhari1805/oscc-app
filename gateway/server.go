package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type Server struct {
	cfg    *Config
	auth   *Auth
	lim    *Limiter
	mailer Mailer
	audit  *Audit
	infer  *InferenceClient
	sem    chan struct{}
	xsem   chan struct{} // explanations are heavy: separate, much smaller pool
	log    *slog.Logger
}

func NewServer(cfg *Config, mailer Mailer, audit *Audit, log *slog.Logger) *Server {
	return &Server{cfg: cfg, auth: NewAuth(cfg), lim: NewLimiter(), mailer: mailer, audit: audit,
		infer: NewInferenceClient(cfg.InferenceURL, cfg.InternalToken),
		sem:   make(chan struct{}, max(1, cfg.MaxConcurrentInfer)), xsem: make(chan struct{}, max(1, cfg.MaxConcurrentExplain)), log: log}
}

func (s *Server) Sweep() { s.auth.Sweep(); s.lim.Sweep() }

// ---------- helpers ----------

type apiError struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	RetryAfter int    `json:"retryAfterSeconds,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]any{"error": apiError{Code: code, Message: msg}})
}

func writeRate(w http.ResponseWriter, retry time.Duration) {
	secs := int(retry.Seconds() + 0.999)
	w.Header().Set("Retry-After", strconv.Itoa(secs))
	writeJSON(w, http.StatusTooManyRequests, map[string]any{"error": apiError{
		Code: "rate_limited", Message: "Too many requests. Please wait before trying again.", RetryAfter: secs}})
}

func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(strings.ToLower(ct), "application/json") {
		writeErr(w, http.StatusUnsupportedMediaType, "unsupported_media_type", "Expected application/json.")
		return false
	}
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil || dec.More() {
		writeErr(w, http.StatusBadRequest, "bad_request", "Malformed request.")
		return false
	}
	return true
}

func (s *Server) clientIP(r *http.Request) string {
	var ip net.IP
	if s.cfg.TrustedProxy { // rightmost entry = the one appended by OUR proxy; left entries are client-controlled
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			parts := strings.Split(xff, ",")
			ip = net.ParseIP(strings.TrimSpace(parts[len(parts)-1]))
		}
	}
	if ip == nil {
		host, _, err := net.SplitHostPort(r.RemoteAddr)
		if err != nil {
			host = r.RemoteAddr
		}
		ip = net.ParseIP(host)
	}
	if ip == nil {
		return "unknown"
	}
	if v4 := ip.To4(); v4 != nil {
		return v4.String()
	}
	return ip.Mask(net.CIDRMask(64, 128)).String() + "/64" // a whole /64 is one client (IPv6 rotation bypass)
}

func (s *Server) ipTag(ip string) string {
	h := hmac.New(sha256.New, s.cfg.SessionSecret)
	h.Write([]byte("ip|" + ip))
	return hex.EncodeToString(h.Sum(nil))[:10]
}

// ---------- middleware ----------

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(c int) { w.status = c; w.ResponseWriter.WriteHeader(c) }

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]bool{"ok": true}) })
	mux.HandleFunc("POST /api/auth/request", s.handleAuthRequest)
	mux.HandleFunc("POST /api/auth/verify", s.handleAuthVerify)
	mux.HandleFunc("POST /api/auth/logout", s.authed(s.handleLogout))
	mux.HandleFunc("GET /api/me", s.authed(s.handleMe))
	mux.HandleFunc("POST /api/predict", s.authed(s.handlePredict))
	mux.HandleFunc("POST /api/explain", s.authed(s.handleExplain))
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) { writeErr(w, 404, "not_found", "Not found.") })
	mux.Handle("/", s.static())
	return s.recoverMW(s.logMW(s.securityMW(s.apiGuardMW(mux))))
}

func (s *Server) recoverMW(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				s.log.Error("panic", "path", r.URL.Path, "err", v)
				writeErr(w, 500, "internal", "Something went wrong.")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func (s *Server) logMW(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(sw, r)
		s.log.Info("req", "m", r.Method, "path", r.URL.Path, "status", sw.status, "ms", time.Since(t0).Milliseconds(), "ip", s.ipTag(s.clientIP(r)))
	})
}

const csp = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; font-src 'self'; " +
	"connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"

func (s *Server) securityMW(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Content-Security-Policy", csp)
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()")
		h.Set("Cross-Origin-Opener-Policy", "same-origin")
		h.Set("Cross-Origin-Resource-Policy", "same-origin")
		if s.cfg.CookieSecure {
			h.Set("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
		}
		next.ServeHTTP(w, r)
	})
}

// apiGuardMW: no-store caching, per-IP global limit, and same-origin enforcement for writes.
func (s *Server) apiGuardMW(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			w.Header().Set("Cache-Control", "no-store")
			ip := s.clientIP(r)
			if ok, retry := s.lim.Allow(Rule{"global-ip", ip, s.cfg.Limits.GlobalPerMinIP, time.Minute}); !ok {
				writeRate(w, retry)
				return
			}
			if r.Method != http.MethodGet && r.Method != http.MethodHead && r.Method != http.MethodOptions {
				if o := r.Header.Get("Origin"); o != "" {
					u, err := url.Parse(o)
					if err != nil || u.Host != r.Host {
						writeErr(w, 403, "bad_origin", "Cross-origin request blocked.")
						return
					}
				}
			}
		}
		next.ServeHTTP(w, r)
	})
}

type authedFn func(w http.ResponseWriter, r *http.Request, sess *Session, token, ip string)

func (s *Server) authed(fn authedFn) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c, err := r.Cookie(s.cfg.CookieName)
		var sess *Session
		if err == nil {
			sess = s.auth.Lookup(c.Value)
		}
		if sess == nil {
			writeErr(w, 401, "unauthenticated", "Please sign in.")
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			if subtle.ConstantTimeCompare([]byte(r.Header.Get("X-CSRF-Token")), []byte(sess.CSRF)) != 1 {
				writeErr(w, 403, "csrf", "Missing or invalid CSRF token.")
				return
			}
		}
		fn(w, r, sess, c.Value, s.clientIP(r))
	}
}

// ---------- auth handlers ----------

var codeRe = regexp.MustCompile(`^[0-9]{6}$`)

func (s *Server) handleAuthRequest(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Email string `json:"email"`
	}
	if !decodeJSON(w, r, &in) {
		return
	}
	email, ok := normalizeEmail(in.Email)
	if !ok {
		writeErr(w, 400, "invalid_email", "Enter a valid e-mail address.")
		return
	}
	ip, L := s.clientIP(r), s.cfg.Limits
	if ok, retry := s.lim.Allow(
		Rule{"otpreq-ip", ip, L.OTPReqPerIPHour, time.Hour},
		Rule{"otpreq-email", email, L.OTPReqPerEmail15m, 15 * time.Minute}); !ok {
		s.audit.Log("auth_request_limited", "", s.ipTag(ip), nil)
		writeRate(w, retry)
		return
	}
	code, issued := s.auth.IssueCode(email)
	if issued {
		go func() { // async: response time must not reveal whether the address is permitted
			if err := s.mailer.SendCode(email, code); err != nil {
				s.log.Error("send code failed", "err", err)
			}
		}()
	}
	if !issued && s.cfg.DevEchoOTP { // dev only (refused in production): the HTTP reply is deliberately identical, so say why nothing arrived
		s.log.Warn("DEV: no code sent because this address is not on the allow-list (or is locked out). Set ALLOWED_EMAILS or ALLOWED_DOMAINS.", "email", email)
	}
	s.audit.Log("auth_request", s.auth.UserID(email), s.ipTag(ip), map[string]any{"issued": issued})
	writeJSON(w, http.StatusAccepted, map[string]string{"message": "If this address is permitted, a 6-digit code has been sent."})
}

func (s *Server) handleAuthVerify(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Email string `json:"email"`
		Code  string `json:"code"`
	}
	if !decodeJSON(w, r, &in) {
		return
	}
	email, ok := normalizeEmail(in.Email)
	if !ok || !codeRe.MatchString(in.Code) {
		writeErr(w, 401, "invalid_or_expired_code", "That code is invalid or has expired.")
		return
	}
	ip, L := s.clientIP(r), s.cfg.Limits
	if ok, retry := s.lim.Allow(
		Rule{"verify-ip", ip, L.VerifyPerIP15m, 15 * time.Minute},
		Rule{"verify-email", email, L.VerifyPerEmail15m, 15 * time.Minute}); !ok {
		s.audit.Log("auth_verify_limited", s.auth.UserID(email), s.ipTag(ip), nil)
		writeRate(w, retry)
		return
	}
	token, sess, err := s.auth.Verify(email, in.Code)
	if err != nil {
		s.audit.Log("auth_fail", s.auth.UserID(email), s.ipTag(ip), nil)
		writeErr(w, 401, "invalid_or_expired_code", "That code is invalid or has expired.")
		return
	}
	http.SetCookie(w, &http.Cookie{Name: s.cfg.CookieName, Value: token, Path: "/", HttpOnly: true,
		Secure: s.cfg.CookieSecure, SameSite: http.SameSiteStrictMode, MaxAge: int(sessionAbsolute.Seconds())})
	s.audit.Log("auth_ok", s.auth.UserID(email), s.ipTag(ip), nil)
	writeJSON(w, 200, map[string]any{"email": sess.Email, "csrfToken": sess.CSRF})
}

func (s *Server) handleMe(w http.ResponseWriter, r *http.Request, sess *Session, _ string, _ string) {
	writeJSON(w, 200, map[string]any{"email": sess.Email, "csrfToken": sess.CSRF})
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request, sess *Session, token, ip string) {
	s.auth.Logout(token)
	http.SetCookie(w, &http.Cookie{Name: s.cfg.CookieName, Value: "", Path: "/", HttpOnly: true,
		Secure: s.cfg.CookieSecure, SameSite: http.SameSiteStrictMode, MaxAge: -1})
	s.audit.Log("logout", s.auth.UserID(sess.Email), s.ipTag(ip), nil)
	writeJSON(w, 200, map[string]bool{"ok": true})
}

// ---------- predict ----------

var qcMessages = map[string]string{
	"LOW_TISSUE":     "Little tissue was detected in this field, so the result may be unreliable.",
	"LOW_RESOLUTION": "This image is smaller than the fields the model was trained on (about 1600×1200).",
}
var rejectMessages = map[string]string{
	"unsupported_format": "Only JPEG and PNG images are supported.",
	"undecodable_image":  "This file could not be read as an image. It may be corrupt or truncated.",
	"image_too_large":    "This image has too many pixels.",
	"bad_dimensions":     "Image dimensions are outside the supported range (128–20000 px per side).",
	"corrupt_image":      "This file could not be read as an image. It may be corrupt or truncated.",
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request, sess *Session, _ string, ip string) {
	uid, L := s.auth.UserID(sess.Email), s.cfg.Limits
	// Rate-limit BEFORE reading the body so abusers cannot make us ingest large uploads.
	if ok, retry := s.lim.Allow(
		Rule{"pred-min", uid, L.PredictPerMin, time.Minute},
		Rule{"pred-hour", uid, L.PredictPerHour, time.Hour},
		Rule{"pred-day", uid, L.PredictPerDay, 24 * time.Hour},
		Rule{"pred-ip", ip, L.PredictPerMinIP, time.Minute}); !ok {
		s.audit.Log("predict_limited", uid, s.ipTag(ip), nil)
		writeRate(w, retry)
		return
	}
	mag, clean, ok := s.readImage(w, r, uid, ip, "predict")
	if !ok {
		return
	}
	select { // bounded concurrency toward the model
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-time.After(3 * time.Second):
		w.Header().Set("Retry-After", "5")
		writeErr(w, 503, "busy", "The analysis service is busy. Please try again shortly.")
		return
	case <-r.Context().Done():
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 80*time.Second)
	defer cancel()
	res, reason, err := s.infer.Predict(ctx, mag, clean)
	if err != nil {
		if errors.Is(err, errInferBadImage) {
			msg, ok := rejectMessages[reason]
			if !ok {
				reason, msg = "undecodable_image", rejectMessages["undecodable_image"]
			}
			writeErr(w, 422, reason, msg)
			return
		}
		s.log.Error("inference failed", "err", err)
		s.audit.Log("predict_error", uid, s.ipTag(ip), nil)
		writeErr(w, 503, "analysis_unavailable", "The analysis service is unavailable. Please try again later.")
		return
	}
	label := "uncertain"
	switch {
	case res.POscc >= s.cfg.HighBand:
		label = "oscc_like"
	case res.POscc <= s.cfg.LowBand:
		label = "normal_like"
	}
	type qcItem struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	qc := []qcItem{}
	for _, c := range res.QC {
		if m, ok := qcMessages[c]; ok {
			qc = append(qc, qcItem{c, m})
		}
	}
	s.audit.Log("predict", uid, s.ipTag(ip), map[string]any{"mag": mag, "label": label, "qc": res.QC, "ms": res.LatencyMS})
	writeJSON(w, 200, map[string]any{
		"resultId": randToken(9), "magnification": mag, "pOscc": res.POscc, "label": label,
		"band":           map[string]float64{"low": s.cfg.LowBand, "high": s.cfg.HighBand},
		"tissueFraction": res.TissueFraction, "qc": qc, "model": res.Model,
		"createdAt": time.Now().UTC().Format(time.RFC3339),
		"notice":    "Research prototype. Not a medical device. Not validated for clinical use.",
	})
}

// readImage parses the multipart upload, validates it and strips metadata. On failure it has already written the
// HTTP error and returns ok=false. Shared by /api/predict and /api/explain so validation cannot drift apart.
func (s *Server) readImage(w http.ResponseWriter, r *http.Request, uid, ip, event string) (mag string, clean []byte, ok bool) {
	if !strings.HasPrefix(strings.ToLower(r.Header.Get("Content-Type")), "multipart/form-data") {
		writeErr(w, 415, "unsupported_media_type", "Expected multipart/form-data.")
		return "", nil, false
	}
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxUploadBytes+(1<<20))
	mr, err := r.MultipartReader()
	if err != nil {
		writeErr(w, 400, "bad_request", "Malformed upload.")
		return "", nil, false
	}
	var img []byte
	for parts := 0; ; parts++ {
		if parts > 4 {
			writeErr(w, 400, "bad_request", "Malformed upload.")
			return "", nil, false
		}
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			s.uploadErr(w, err)
			return "", nil, false
		}
		switch part.FormName() {
		case "magnification":
			b, _ := io.ReadAll(io.LimitReader(part, 16))
			mag = strings.TrimSpace(string(b))
		case "image":
			img, err = io.ReadAll(io.LimitReader(part, s.cfg.MaxUploadBytes+1))
			if err != nil {
				s.uploadErr(w, err)
				return "", nil, false
			}
			if int64(len(img)) > s.cfg.MaxUploadBytes {
				writeErr(w, 413, "too_large", "File is too large (limit "+strconv.FormatInt(s.cfg.MaxUploadBytes>>20, 10)+" MB).")
				return "", nil, false
			}
		default:
			_, _ = io.Copy(io.Discard, io.LimitReader(part, 1024))
		}
	}
	if mag != "100x" && mag != "400x" {
		writeErr(w, 400, "bad_magnification", "Choose 100x or 400x.")
		return "", nil, false
	}
	if len(img) == 0 {
		writeErr(w, 400, "no_image", "Attach an image.")
		return "", nil, false
	}
	clean, _, _, _, err = sanitizeImage(img)
	if err != nil {
		st := 422
		if errors.Is(err, errUnsupported) {
			st = 415
		}
		s.audit.Log(event+"_rejected", uid, s.ipTag(ip), map[string]any{"reason": err.Error()})
		writeErr(w, st, err.Error(), rejectMessages[err.Error()])
		return "", nil, false
	}
	return mag, clean, true
}

// explainWait is how long a request waits for the single explanation slot (a var so tests can shorten it).
var explainWait = 3 * time.Second

// handleExplain returns an occlusion-sensitivity map for one image. It repeats the full validation and sanitising
// of /api/predict, but has its own rate limits and concurrency gate because each call costs ~30 model passes.
func (s *Server) handleExplain(w http.ResponseWriter, r *http.Request, sess *Session, _ string, ip string) {
	uid, L := s.auth.UserID(sess.Email), s.cfg.Limits
	if ok, retry := s.lim.Allow(
		Rule{"exp-min", uid, L.ExplainPerMin, time.Minute},
		Rule{"exp-hour", uid, L.ExplainPerHour, time.Hour},
		Rule{"exp-day", uid, L.ExplainPerDay, 24 * time.Hour},
		Rule{"exp-ip", ip, L.ExplainPerMinIP, time.Minute}); !ok {
		s.audit.Log("explain_limited", uid, s.ipTag(ip), nil)
		writeRate(w, retry)
		return
	}
	mag, clean, ok := s.readImage(w, r, uid, ip, "explain")
	if !ok {
		return
	}
	select {
	case s.xsem <- struct{}{}:
		defer func() { <-s.xsem }()
	case <-time.After(explainWait):
		w.Header().Set("Retry-After", "10")
		writeErr(w, 503, "busy", "Another explanation is being computed. Please try again in a few seconds.")
		return
	case <-r.Context().Done():
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()
	res, reason, err := s.infer.Explain(ctx, mag, clean)
	if err != nil {
		switch {
		case errors.Is(err, errInferBadImage):
			msg, ok := rejectMessages[reason]
			if !ok {
				reason, msg = "undecodable_image", rejectMessages["undecodable_image"]
			}
			writeErr(w, 422, reason, msg)
		case errors.Is(err, errInferBusy):
			w.Header().Set("Retry-After", "10")
			writeErr(w, 503, "busy", "The analysis service is busy. Please try again in a few seconds.")
		default:
			s.log.Error("explain failed", "err", err)
			s.audit.Log("explain_error", uid, s.ipTag(ip), nil)
			writeErr(w, 503, "analysis_unavailable", "The analysis service is unavailable. Please try again later.")
		}
		return
	}
	c := res.Check
	s.audit.Log("explain", uid, s.ipTag(ip), map[string]any{"mag": mag, "supported": c.Supported, "ms": res.LatencyMS})
	writeJSON(w, 200, map[string]any{
		"magnification": mag, "pOscc": res.POscc, "method": res.Method, "passes": res.Passes,
		"grid": res.Grid, "windowCells": res.Window, "relative": res.Relative, "maxRel": res.MaxRel, "maxAbs": res.MaxAbs,
		"check": map[string]any{
			"cellsHidden": c.CellsHidden, "direction": c.Direction, "pBefore": c.PBefore, "pTopHidden": c.PTopHidden,
			"pRandomHidden": c.PRandomHidden, "deltaTop": c.DeltaTop, "deltaRandom": c.DeltaRandom, "supported": c.Supported,
		},
		"model": res.Model, "createdAt": time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) uploadErr(w http.ResponseWriter, err error) {
	var mbe *http.MaxBytesError
	if errors.As(err, &mbe) {
		writeErr(w, 413, "too_large", "File is too large (limit "+strconv.FormatInt(s.cfg.MaxUploadBytes>>20, 10)+" MB).")
		return
	}
	writeErr(w, 400, "bad_request", "Malformed upload.")
}

// ---------- static (SPA) ----------

func (s *Server) static() http.Handler {
	root, _ := filepath.Abs(s.cfg.WebDir)
	serve := func(w http.ResponseWriter, r *http.Request, name string, immutable bool) {
		f, err := os.Open(name)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		defer f.Close()
		st, _ := f.Stat()
		if immutable {
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		} else {
			w.Header().Set("Cache-Control", "no-cache")
		}
		http.ServeContent(w, r, name, st.ModTime(), f)
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			writeErr(w, 405, "method_not_allowed", "Method not allowed.")
			return
		}
		clean := path.Clean("/" + r.URL.Path)
		if clean == "/favicon.ico" { // the app ships no favicon; answer empty instead of the SPA page
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if !strings.Contains(clean, "/.") { // never serve dotfiles
			p := filepath.Join(root, filepath.FromSlash(clean))
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() && strings.HasPrefix(p, root) {
				serve(w, r, p, strings.HasPrefix(clean, "/assets/"))
				return
			}
		}
		serve(w, r, filepath.Join(root, "index.html"), false) // SPA fallback
	})
}
