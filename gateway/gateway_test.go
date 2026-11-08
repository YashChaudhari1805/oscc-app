package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ---------- fixtures ----------

func gradient(w, h int) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			im.Set(x, y, color.RGBA{uint8(x), uint8(y), uint8(x + y), 255})
		}
	}
	return im
}

func jpegBytes(t *testing.T, w, h int) []byte {
	var b bytes.Buffer
	if err := jpeg.Encode(&b, gradient(w, h), &jpeg.Options{Quality: 90}); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func pngBytes(t *testing.T, w, h int) []byte {
	var b bytes.Buffer
	if err := png.Encode(&b, gradient(w, h)); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func injectJPEGSegment(b []byte, marker byte, payload []byte) []byte {
	seg := []byte{0xFF, marker, 0, 0}
	binary.BigEndian.PutUint16(seg[2:], uint16(len(payload)+2))
	seg = append(seg, payload...)
	out := append([]byte{}, b[:2]...)
	out = append(out, seg...)
	return append(out, b[2:]...)
}

func pngChunk(typ string, data []byte) []byte {
	out := make([]byte, 8, 12+len(data))
	binary.BigEndian.PutUint32(out, uint32(len(data)))
	copy(out[4:], typ)
	out = append(out, data...)
	crc := crc32.ChecksumIEEE(out[4:])
	return binary.BigEndian.AppendUint32(out, crc)
}

func injectPNGChunkBeforeIDAT(b []byte, chunk []byte) []byte {
	i := bytes.Index(b, []byte("IDAT")) - 4
	return append(append(append([]byte{}, b[:i]...), chunk...), b[i:]...)
}

type capMailer struct{ ch chan [2]string }

func (m capMailer) SendCode(to, code string) error { m.ch <- [2]string{to, code}; return nil }

type env struct {
	srv    *Server
	ts     *httptest.Server
	mail   capMailer
	infer  *httptest.Server
	got    chan []byte
	client *http.Client
	cfg    *Config
}

func newEnv(t *testing.T, tweak func(*Config)) *env {
	t.Helper()
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "index.html"), []byte("<html>spa</html>"), 0o644)
	os.WriteFile(filepath.Join(dir, ".secret"), []byte("TOPSECRET"), 0o644)
	os.Mkdir(filepath.Join(dir, "assets"), 0o755)
	os.WriteFile(filepath.Join(dir, "assets", "app.js"), []byte("console.log(1)"), 0o644)
	e := &env{got: make(chan []byte, 8), mail: capMailer{make(chan [2]string, 8)}}
	e.infer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Internal-Token") != strings.Repeat("i", 32) {
			w.WriteHeader(401)
			return
		}
		b, _ := io.ReadAll(r.Body)
		e.got <- b
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"p_oscc":0.93,"mag":"`+r.URL.Query().Get("mag")+`","tissue_fraction":0.8,"original_size":[640,480],"qc":["LOW_RESOLUTION"],"model":{"version":"100x-test","backend":"onnx","img_hw":[288,384]},"latency_ms":12}`)
	}))
	cfg := &Config{
		Env: "development", InferenceURL: e.infer.URL, InternalToken: strings.Repeat("i", 32),
		SessionSecret: []byte(strings.Repeat("s", 32)), AllowedEmails: map[string]bool{"doc@hospital.org": true},
		AllowedDomains: map[string]bool{"clinic.org": true}, CookieName: "oscc_session", WebDir: dir,
		MaxUploadBytes: 2 << 20, MaxConcurrentInfer: 2, LowBand: 0.2, HighBand: 0.8,
		Limits: Limits{GlobalPerMinIP: 1000, OTPReqPerIPHour: 10, OTPReqPerEmail15m: 3, VerifyPerIP15m: 50, VerifyPerEmail15m: 50,
			PredictPerMin: 5, PredictPerHour: 40, PredictPerDay: 150, PredictPerMinIP: 15},
	}
	if tweak != nil {
		tweak(cfg)
	}
	audit, _ := NewAudit(filepath.Join(dir, "audit.jsonl"))
	e.cfg = cfg
	e.srv = NewServer(cfg, e.mail, audit, slog.New(slog.NewTextHandler(io.Discard, nil)))
	e.ts = httptest.NewServer(e.srv.Handler())
	jar, _ := cookiejar.New(nil)
	e.client = &http.Client{Jar: jar, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	t.Cleanup(func() { e.ts.Close(); e.infer.Close() })
	return e
}

func (e *env) postJSON(path string, v any, hdr map[string]string) (*http.Response, []byte) {
	b, _ := json.Marshal(v)
	req, _ := http.NewRequest("POST", e.ts.URL+path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	for k, val := range hdr {
		req.Header.Set(k, val)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		panic(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, body
}

func (e *env) get(path string) (*http.Response, []byte) {
	resp, err := e.client.Get(e.ts.URL + path)
	if err != nil {
		panic(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, body
}

func (e *env) login(t *testing.T, email string) string {
	t.Helper()
	if r, _ := e.postJSON("/api/auth/request", map[string]string{"email": email}, nil); r.StatusCode != 202 {
		t.Fatalf("request code: %d", r.StatusCode)
	}
	var m [2]string
	select {
	case m = <-e.mail.ch:
	case <-time.After(2 * time.Second):
		t.Fatal("no code mailed")
	}
	r, body := e.postJSON("/api/auth/verify", map[string]string{"email": email, "code": m[1]}, nil)
	if r.StatusCode != 200 {
		t.Fatalf("verify: %d %s", r.StatusCode, body)
	}
	var out struct {
		CSRFToken string `json:"csrfToken"`
	}
	json.Unmarshal(body, &out)
	return out.CSRFToken
}

func (e *env) predict(csrf, mag string, img []byte) (*http.Response, []byte) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("magnification", mag)
	if img != nil {
		fw, _ := mw.CreateFormFile("image", "../../evil name.jpg")
		fw.Write(img)
	}
	mw.Close()
	req, _ := http.NewRequest("POST", e.ts.URL+"/api/predict", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	if csrf != "" {
		req.Header.Set("X-CSRF-Token", csrf)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		panic(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, body
}

// ---------- unit: limiter ----------

func TestLimiterSlidingWindowAndAtomicity(t *testing.T) {
	l := NewLimiter()
	now := time.Unix(1000, 0)
	l.now = func() time.Time { return now }
	a := Rule{"a", "k", 2, time.Minute}
	b := Rule{"b", "k", 1, time.Hour}
	if ok, _ := l.Allow(a, b); !ok {
		t.Fatal("first should pass")
	}
	if ok, retry := l.Allow(a, b); ok || retry < time.Minute { // b denies; retry must reflect b's window
		t.Fatalf("second must be denied by b: ok=%v retry=%v", ok, retry)
	}
	// a must NOT have been consumed by the denied call (atomic): a still has 1 slot left
	if ok, _ := l.Allow(a); !ok {
		t.Fatal("rule a was consumed by a denied request")
	}
	now = now.Add(61 * time.Second)
	if ok, _ := l.Allow(a); !ok {
		t.Fatal("window should have slid")
	}
	now = now.Add(2 * time.Hour)
	l.Sweep()
	if len(l.hits) != 0 {
		t.Fatalf("sweep left %d buckets", len(l.hits))
	}
}

func TestLimiterFailsClosedAtKeyCap(t *testing.T) {
	l := NewLimiter()
	l.maxKeys = 3
	for i := 0; i < 3; i++ {
		l.Allow(Rule{"x", fmt.Sprint(i), 5, time.Minute})
	}
	if ok, _ := l.Allow(Rule{"x", "new", 5, time.Minute}); ok {
		t.Fatal("must fail closed when key cap reached")
	}
}

// ---------- unit: scrubbing ----------

func TestScrubJPEGRemovesMetadataKeepsPixels(t *testing.T) {
	orig := jpegBytes(t, 256, 256)
	dirty := injectJPEGSegment(orig, 0xE1, []byte("Exif\x00\x00PATIENT-JOHN-DOE GPS 19.99N"))
	dirty = injectJPEGSegment(dirty, 0xFE, []byte("comment MRN123456"))
	if !bytes.Contains(dirty, []byte("PATIENT-JOHN-DOE")) {
		t.Fatal("fixture broken")
	}
	clean, kind, w, h, err := sanitizeImage(dirty)
	if err != nil || kind != "jpeg" || w != 256 || h != 256 {
		t.Fatalf("sanitize: %v %s %d %d", err, kind, w, h)
	}
	if bytes.Contains(clean, []byte("PATIENT-JOHN-DOE")) || bytes.Contains(clean, []byte("MRN123456")) {
		t.Fatal("metadata survived")
	}
	a, _ := jpeg.Decode(bytes.NewReader(orig))
	b, err := jpeg.Decode(bytes.NewReader(clean))
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 256; y += 7 {
		for x := 0; x < 256; x += 5 {
			if a.At(x, y) != b.At(x, y) {
				t.Fatal("pixels changed by scrubbing")
			}
		}
	}
}

func TestScrubPNGRemovesTextChunksVerifiesCRC(t *testing.T) {
	orig := pngBytes(t, 200, 200)
	dirty := injectPNGChunkBeforeIDAT(orig, pngChunk("tEXt", []byte("Author\x00Dr Patient Name MRN-998")))
	dirty = injectPNGChunkBeforeIDAT(dirty, pngChunk("eXIf", []byte("exifdata-GPS")))
	clean, kind, _, _, err := sanitizeImage(dirty)
	if err != nil || kind != "png" {
		t.Fatal(err)
	}
	if bytes.Contains(clean, []byte("MRN-998")) || bytes.Contains(clean, []byte("exifdata")) {
		t.Fatal("PNG metadata survived")
	}
	a, _ := png.Decode(bytes.NewReader(orig))
	b, err := png.Decode(bytes.NewReader(clean))
	if err != nil || a.At(10, 10) != b.At(10, 10) {
		t.Fatal("png pixels changed")
	}
	bad := append([]byte{}, orig...)
	bad[40] ^= 0xFF // corrupt a byte inside a chunk -> CRC mismatch
	if _, _, _, _, err := sanitizeImage(bad); err == nil {
		t.Fatal("corrupt PNG accepted")
	}
}

func TestSanitizeRejects(t *testing.T) {
	cases := map[string][]byte{
		"gif":   []byte("GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
		"pdf":   []byte("%PDF-1.7 ..."),
		"html":  []byte("<script>alert(1)</script>"),
		"empty": {},
		"svg":   []byte("<svg xmlns='http://www.w3.org/2000/svg'/>"),
	}
	for name, b := range cases {
		if _, _, _, _, err := sanitizeImage(b); err != errUnsupported {
			t.Errorf("%s: got %v want unsupported", name, err)
		}
	}
	j := jpegBytes(t, 256, 256)
	if _, _, _, _, err := sanitizeImage(j[:60]); err == nil {
		t.Error("truncated jpeg accepted")
	}
	if _, _, _, _, err := sanitizeImage(jpegBytes(t, 64, 64)); err != errDimensions {
		t.Errorf("small image: %v", err)
	}
	// polyglot: JPEG magic followed by junk must not be accepted
	if _, _, _, _, err := sanitizeImage(append([]byte{0xFF, 0xD8, 0xFF}, []byte("<html>")...)); err == nil {
		t.Error("polyglot accepted")
	}
}

// ---------- integration ----------

func TestUnauthenticatedAndSecurityHeaders(t *testing.T) {
	e := newEnv(t, nil)
	if r, _ := e.get("/api/me"); r.StatusCode != 401 {
		t.Fatalf("me: %d", r.StatusCode)
	}
	if r, _ := e.predict("", "100x", jpegBytes(t, 256, 256)); r.StatusCode != 401 {
		t.Fatalf("predict unauth: %d", r.StatusCode)
	}
	r, _ := e.get("/")
	for _, h := range []string{"Content-Security-Policy", "X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy", "Permissions-Policy"} {
		if r.Header.Get(h) == "" {
			t.Errorf("missing %s", h)
		}
	}
	if !strings.Contains(r.Header.Get("Content-Security-Policy"), "frame-ancestors 'none'") {
		t.Error("CSP lacks frame-ancestors")
	}
	if r, _ := e.get("/api/health"); r.Header.Get("Cache-Control") != "no-store" {
		t.Error("API must be no-store")
	}
}

func TestEnumerationResistanceAndAllowlist(t *testing.T) {
	e := newEnv(t, nil)
	r1, b1 := e.postJSON("/api/auth/request", map[string]string{"email": "stranger@evil.com"}, nil)
	r2, b2 := e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	if r1.StatusCode != 202 || r2.StatusCode != 202 || string(b1) != string(b2) {
		t.Fatalf("responses must be identical: %d/%d %s | %s", r1.StatusCode, r2.StatusCode, b1, b2)
	}
	select {
	case m := <-e.mail.ch:
		if m[0] != "doc@hospital.org" {
			t.Fatalf("mailed %s", m[0])
		}
	case <-time.After(time.Second):
		t.Fatal("allowed address got no code")
	}
	select {
	case m := <-e.mail.ch:
		t.Fatalf("non-allowed address received a code: %v", m)
	case <-time.After(200 * time.Millisecond):
	}
	// domain allow-list works; invalid formats rejected
	e.login(t, "Nurse@Clinic.org")
	for _, bad := range []string{"", "no-at", "a@b@c.com", "Name <x@clinic.org>", "x@clinic.org,y@evil.com", strings.Repeat("a", 300) + "@clinic.org"} {
		if r, _ := e.postJSON("/api/auth/request", map[string]string{"email": bad}, nil); r.StatusCode != 400 {
			t.Errorf("%q accepted: %d", bad, r.StatusCode)
		}
	}
}

func TestCodeBruteForceLockout(t *testing.T) {
	e := newEnv(t, nil)
	e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	m := <-e.mail.ch
	wrong := "000000"
	if m[1] == wrong {
		wrong = "111111"
	}
	for i := 0; i < maxCodeAttempts; i++ {
		if r, _ := e.postJSON("/api/auth/verify", map[string]string{"email": "doc@hospital.org", "code": wrong}, nil); r.StatusCode != 401 {
			t.Fatalf("attempt %d: %d", i, r.StatusCode)
		}
	}
	// the CORRECT code must now be dead, and re-requesting during lockout must not mint a new one
	if r, _ := e.postJSON("/api/auth/verify", map[string]string{"email": "doc@hospital.org", "code": m[1]}, nil); r.StatusCode != 401 {
		t.Fatal("correct code accepted after lockout")
	}
	e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	select {
	case <-e.mail.ch:
		t.Fatal("code issued during lockout")
	case <-time.After(200 * time.Millisecond):
	}
}

func TestCodeIsSingleUseAndRequestRateLimited(t *testing.T) {
	e := newEnv(t, nil)
	e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	m := <-e.mail.ch
	if r, _ := e.postJSON("/api/auth/verify", map[string]string{"email": m[0], "code": m[1]}, nil); r.StatusCode != 200 {
		t.Fatal("first use should work")
	}
	if r, _ := e.postJSON("/api/auth/verify", map[string]string{"email": m[0], "code": m[1]}, nil); r.StatusCode != 401 {
		t.Fatal("code reused")
	}
	for i := 0; i < 2; i++ { // 1 already made; limit is 3 per email per 15m
		e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	}
	r, body := e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	if r.StatusCode != 429 || r.Header.Get("Retry-After") == "" {
		t.Fatalf("expected 429 with Retry-After, got %d %s", r.StatusCode, body)
	}
	// limit also applies to non-allowed addresses (no oracle, no mail-bombing)
	for i := 0; i < 3; i++ {
		e.postJSON("/api/auth/request", map[string]string{"email": "x@evil.com"}, nil)
	}
	if r, _ := e.postJSON("/api/auth/request", map[string]string{"email": "x@evil.com"}, nil); r.StatusCode != 429 {
		t.Fatalf("non-allowed address not rate limited: %d", r.StatusCode)
	}
}

func TestSessionCookieAndCSRF(t *testing.T) {
	e := newEnv(t, nil)
	csrf := e.login(t, "doc@hospital.org")
	u := e.ts.URL
	var ck *http.Cookie
	for _, c := range e.client.Jar.Cookies(mustURL(u)) {
		if c.Name == "oscc_session" {
			ck = c
		}
	}
	if ck == nil {
		t.Fatal("no session cookie")
	}
	// attributes are only visible on the Set-Cookie header
	r, _ := e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	_ = r
	e2 := newEnv(t, nil)
	e2.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, nil)
	m := <-e2.mail.ch
	resp, _ := e2.postJSON("/api/auth/verify", map[string]string{"email": m[0], "code": m[1]}, nil)
	sc := resp.Header.Get("Set-Cookie")
	if !strings.Contains(sc, "HttpOnly") || !strings.Contains(sc, "SameSite=Strict") || !strings.Contains(sc, "Path=/") {
		t.Fatalf("weak cookie: %s", sc)
	}
	if r, _ := e.predict("", "100x", jpegBytes(t, 256, 256)); r.StatusCode != 403 {
		t.Fatalf("predict without CSRF: %d", r.StatusCode)
	}
	if r, _ := e.predict("wrong-token", "100x", jpegBytes(t, 256, 256)); r.StatusCode != 403 {
		t.Fatalf("predict with bad CSRF: %d", r.StatusCode)
	}
	if r, _ := e.postJSON("/api/auth/logout", struct{}{}, nil); r.StatusCode != 403 {
		t.Fatalf("logout without CSRF: %d", r.StatusCode)
	}
	r, _ = e.postJSON("/api/auth/logout", struct{}{}, map[string]string{"X-CSRF-Token": csrf})
	if r.StatusCode != 200 {
		t.Fatalf("logout: %d", r.StatusCode)
	}
	if r, _ := e.get("/api/me"); r.StatusCode != 401 {
		t.Fatal("session survived logout")
	}
}

func mustURL(s string) *urlT { u, _ := parseURL(s); return u }

func TestCrossOriginWriteBlocked(t *testing.T) {
	e := newEnv(t, nil)
	r, _ := e.postJSON("/api/auth/request", map[string]string{"email": "doc@hospital.org"}, map[string]string{"Origin": "https://evil.example"})
	if r.StatusCode != 403 {
		t.Fatalf("cross-origin POST not blocked: %d", r.StatusCode)
	}
}

func TestPredictHappyPathScrubsBeforeForwarding(t *testing.T) {
	e := newEnv(t, nil)
	csrf := e.login(t, "doc@hospital.org")
	dirty := injectJPEGSegment(jpegBytes(t, 400, 300), 0xE1, []byte("Exif\x00\x00PATIENT-JOHN-DOE"))
	r, body := e.predict(csrf, "100x", dirty)
	if r.StatusCode != 200 {
		t.Fatalf("%d %s", r.StatusCode, body)
	}
	var out map[string]any
	json.Unmarshal(body, &out)
	if out["label"] != "oscc_like" || out["pOscc"].(float64) != 0.93 || out["magnification"] != "100x" {
		t.Fatalf("bad result: %s", body)
	}
	if qc := out["qc"].([]any); len(qc) != 1 || qc[0].(map[string]any)["code"] != "LOW_RESOLUTION" {
		t.Fatalf("qc missing: %s", body)
	}
	if !strings.Contains(out["notice"].(string), "Not a medical device") {
		t.Fatal("disclaimer missing")
	}
	fwd := <-e.got
	if bytes.Contains(fwd, []byte("PATIENT-JOHN-DOE")) {
		t.Fatal("metadata reached the inference service")
	}
	// audit log must not contain e-mail, filename or image
	a, _ := os.ReadFile(filepath.Join(e.cfg.WebDir, "audit.jsonl"))
	if bytes.Contains(a, []byte("hospital.org")) || bytes.Contains(a, []byte("evil name")) {
		t.Fatalf("audit leaks identifiers: %s", a)
	}
	if !bytes.Contains(a, []byte(`"event":"predict"`)) {
		t.Fatal("predict not audited")
	}
}

func TestPredictValidation(t *testing.T) {
	// invalid uploads deliberately consume rate-limit quota (anti-probing), so lift it for this test
	e := newEnv(t, func(c *Config) { c.Limits.PredictPerMin, c.Limits.PredictPerMinIP = 50, 50 })
	csrf := e.login(t, "doc@hospital.org")
	if r, _ := e.predict(csrf, "40x", jpegBytes(t, 256, 256)); r.StatusCode != 400 {
		t.Errorf("bad mag: %d", r.StatusCode)
	}
	if r, _ := e.predict(csrf, "100x", nil); r.StatusCode != 400 {
		t.Errorf("no image: %d", r.StatusCode)
	}
	if r, _ := e.predict(csrf, "100x", []byte("<html>not an image</html>")); r.StatusCode != 415 {
		t.Errorf("html upload: %d", r.StatusCode)
	}
	if r, _ := e.predict(csrf, "100x", jpegBytes(t, 64, 64)); r.StatusCode != 422 {
		t.Errorf("tiny image: %d", r.StatusCode)
	}
	huge := append(jpegBytes(t, 256, 256), make([]byte, 3<<20)...) // > 2 MB test cap
	if r, _ := e.predict(csrf, "100x", huge); r.StatusCode != 413 {
		t.Errorf("oversize: %d", r.StatusCode)
	}
	// text/plain instead of multipart
	req, _ := http.NewRequest("POST", e.ts.URL+"/api/predict", strings.NewReader("hello"))
	req.Header.Set("X-CSRF-Token", csrf)
	req.Header.Set("Content-Type", "text/plain")
	resp, _ := e.client.Do(req)
	if resp.StatusCode != 415 {
		t.Errorf("non-multipart: %d", resp.StatusCode)
	}
}

func TestPredictRateLimit(t *testing.T) {
	e := newEnv(t, nil)
	csrf := e.login(t, "doc@hospital.org")
	img := jpegBytes(t, 256, 256)
	for i := 0; i < 5; i++ {
		if r, b := e.predict(csrf, "400x", img); r.StatusCode != 200 {
			t.Fatalf("req %d: %d %s", i, r.StatusCode, b)
		}
		<-e.got
	}
	r, body := e.predict(csrf, "400x", img)
	if r.StatusCode != 429 || r.Header.Get("Retry-After") == "" || !strings.Contains(string(body), "rate_limited") {
		t.Fatalf("expected 429, got %d %s", r.StatusCode, body)
	}
	select {
	case <-e.got:
		t.Fatal("rate-limited request still reached inference")
	case <-time.After(150 * time.Millisecond):
	}
}

func TestGlobalIPLimit(t *testing.T) {
	e := newEnv(t, func(c *Config) { c.Limits.GlobalPerMinIP = 5 })
	var last int
	for i := 0; i < 8; i++ {
		r, _ := e.get("/api/health")
		last = r.StatusCode
	}
	if last != 429 {
		t.Fatalf("global limit not enforced: %d", last)
	}
}

func TestInferenceDownAndBadScore(t *testing.T) {
	e := newEnv(t, nil)
	csrf := e.login(t, "doc@hospital.org")
	e.infer.Close()
	r, body := e.predict(csrf, "100x", jpegBytes(t, 256, 256))
	if r.StatusCode != 503 || strings.Contains(string(body), "127.0.0.1") {
		t.Fatalf("down: %d %s", r.StatusCode, body)
	}
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, `{"p_oscc":1.7,"qc":[]}`) }))
	defer bad.Close()
	e.srv.infer = NewInferenceClient(bad.URL, strings.Repeat("i", 32))
	if r, _ := e.predict(csrf, "100x", jpegBytes(t, 256, 256)); r.StatusCode != 503 {
		t.Fatalf("out-of-range score must not be shown: %d", r.StatusCode)
	}
}

func TestStaticServingIsSafe(t *testing.T) {
	e := newEnv(t, nil)
	if r, b := e.get("/"); r.StatusCode != 200 || !strings.Contains(string(b), "spa") {
		t.Fatal("index")
	}
	if r, b := e.get("/some/client/route"); r.StatusCode != 200 || !strings.Contains(string(b), "spa") {
		t.Fatal("spa fallback")
	}
	if r, _ := e.get("/assets/app.js"); !strings.Contains(r.Header.Get("Cache-Control"), "immutable") {
		t.Fatal("assets should be cached immutably")
	}
	for _, p := range []string{"/.secret", "/../../../etc/passwd", "/%2e%2e/%2e%2e/etc/passwd", "/assets/../.secret", "/..%2f..%2fetc/passwd"} {
		_, b := e.get(p)
		if strings.Contains(string(b), "TOPSECRET") || strings.Contains(string(b), "root:") {
			t.Errorf("leak via %s", p)
		}
	}
	if r, _ := e.get("/api/does-not-exist"); r.StatusCode != 404 || !strings.Contains(r.Header.Get("Content-Type"), "json") {
		t.Errorf("unknown api route: %d", r.StatusCode)
	}
}

func TestProductionConfigFailsClosed(t *testing.T) {
	base := map[string]string{"APP_ENV": "production", "INTERNAL_TOKEN": strings.Repeat("i", 32), "SESSION_SECRET": strings.Repeat("s", 40),
		"SMTP_HOST": "smtp.example.com", "SMTP_FROM": "a@b.org", "ALLOWED_DOMAINS": "hospital.org"}
	set := func(over map[string]string) error {
		for _, k := range []string{"APP_ENV", "INTERNAL_TOKEN", "SESSION_SECRET", "SMTP_HOST", "SMTP_FROM", "ALLOWED_DOMAINS", "ALLOWED_EMAILS", "ALLOW_ANY_EMAIL", "DEV_ECHO_OTP", "COOKIE_SECURE"} {
			t.Setenv(k, "")
		}
		for k, v := range base {
			t.Setenv(k, v)
		}
		for k, v := range over {
			t.Setenv(k, v)
		}
		_, err := LoadConfig()
		return err
	}
	if err := set(nil); err != nil {
		t.Fatalf("valid prod config rejected: %v", err)
	}
	for name, over := range map[string]map[string]string{
		"any email":       {"ALLOW_ANY_EMAIL": "true"},
		"dev otp":         {"DEV_ECHO_OTP": "true"},
		"insecure cookie": {"COOKIE_SECURE": "false"},
		"no smtp":         {"SMTP_HOST": ""},
		"no allowlist":    {"ALLOWED_DOMAINS": ""},
		"weak secret":     {"SESSION_SECRET": "short"},
		"weak token":      {"INTERNAL_TOKEN": "short"},
	} {
		if err := set(over); err == nil {
			t.Errorf("%s: insecure production config accepted", name)
		}
	}
}
