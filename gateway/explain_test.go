package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func explainJSON(mut func(m map[string]any)) string {
	rel := make([][]float64, 9)
	for r := range rel {
		rel[r] = make([]float64, 12)
		for c := range rel[r] {
			rel[r][c] = float64(r-4) * 0.1
		}
	}
	m := map[string]any{
		"p_oscc": 0.93, "grid": []int{9, 12}, "window": []int{3, 3}, "relative": rel, "max_abs": 1.2, "max_rel": 0.4,
		"check": map[string]any{"cells_hidden": 16, "direction": "oscc", "p_before": 0.93, "p_top_hidden": 0.41,
			"p_random_hidden": 0.88, "delta_top": 2.1, "delta_random": 0.2, "supported": true},
		"method": "occlusion", "passes": 30, "tissue_fraction": 0.8, "model": map[string]any{"version": "100x-test"}, "latency_ms": 3000,
	}
	if mut != nil {
		mut(m)
	}
	b, _ := json.Marshal(m)
	return string(b)
}

// fakeInference serves canned /v1/explain and records the bodies it receives.
func fakeInference(t *testing.T, payload string, got chan []byte) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Internal-Token") != strings.Repeat("i", 32) {
			w.WriteHeader(401)
			return
		}
		b, _ := io.ReadAll(r.Body)
		if got != nil {
			got <- b
		}
		w.Header().Set("Content-Type", "application/json")
		if strings.HasPrefix(r.URL.Path, "/v1/explain") {
			fmt.Fprint(w, payload)
			return
		}
		fmt.Fprint(w, `{"p_oscc":0.93,"mag":"100x","tissue_fraction":0.8,"original_size":[640,480],"qc":[],"model":{"version":"v"},"latency_ms":5}`)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (e *env) explain(csrf, mag string, img []byte) (*http.Response, []byte) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("magnification", mag)
	if img != nil {
		fw, _ := mw.CreateFormFile("image", "x.jpg")
		fw.Write(img)
	}
	mw.Close()
	req, _ := http.NewRequest("POST", e.ts.URL+"/api/explain", &buf)
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

func (e *env) useInference(t *testing.T, payload string, got chan []byte) {
	e.srv.infer = NewInferenceClient(fakeInference(t, payload, got).URL, strings.Repeat("i", 32))
}

func TestExplainHappyPathScrubsAndAudits(t *testing.T) {
	e := newEnv(t, nil)
	got := make(chan []byte, 4)
	e.useInference(t, explainJSON(nil), got)
	csrf := e.login(t, "doc@hospital.org")
	dirty := injectJPEGSegment(jpegBytes(t, 400, 300), 0xE1, []byte("Exif\x00\x00PATIENT-JOHN-DOE"))
	r, body := e.explain(csrf, "100x", dirty)
	if r.StatusCode != 200 {
		t.Fatalf("%d %s", r.StatusCode, body)
	}
	var out map[string]any
	json.Unmarshal(body, &out)
	grid := out["grid"].([]any)
	if grid[0].(float64) != 9 || grid[1].(float64) != 12 || len(out["relative"].([]any)) != 9 || out["method"] != "occlusion" {
		t.Fatalf("bad payload: %s", body)
	}
	if c := out["check"].(map[string]any); c["supported"] != true || c["direction"] != "oscc" || c["cellsHidden"].(float64) != 16 {
		t.Fatalf("bad check: %v", c)
	}
	if bytes.Contains(<-got, []byte("PATIENT-JOHN-DOE")) {
		t.Fatal("metadata reached the inference service")
	}
	a, _ := os.ReadFile(filepath.Join(e.cfg.WebDir, "audit.jsonl"))
	if !bytes.Contains(a, []byte(`"event":"explain"`)) || bytes.Contains(a, []byte("hospital.org")) {
		t.Fatalf("audit wrong: %s", a)
	}
}

func TestExplainRequiresSessionAndCSRF(t *testing.T) {
	e := newEnv(t, nil)
	e.useInference(t, explainJSON(nil), nil)
	if r, _ := e.explain("", "100x", jpegBytes(t, 256, 256)); r.StatusCode != 401 {
		t.Fatalf("unauthenticated: %d", r.StatusCode)
	}
	csrf := e.login(t, "doc@hospital.org")
	_ = csrf
	if r, _ := e.explain("", "100x", jpegBytes(t, 256, 256)); r.StatusCode != 403 {
		t.Fatalf("no csrf: %d", r.StatusCode)
	}
	if r, _ := e.explain("wrong", "100x", jpegBytes(t, 256, 256)); r.StatusCode != 403 {
		t.Fatalf("bad csrf: %d", r.StatusCode)
	}
}

func TestExplainHasItsOwnRateLimit(t *testing.T) {
	e := newEnv(t, nil) // explain 3/min in this config; predict is 5/min
	e.cfg.Limits.ExplainPerMin, e.cfg.Limits.ExplainPerMinIP = 3, 50
	e.useInference(t, explainJSON(nil), nil)
	csrf := e.login(t, "doc@hospital.org")
	img := jpegBytes(t, 256, 256)
	for i := 0; i < 3; i++ {
		if r, b := e.explain(csrf, "100x", img); r.StatusCode != 200 {
			t.Fatalf("explain %d: %d %s", i, r.StatusCode, b)
		}
	}
	r, body := e.explain(csrf, "100x", img)
	if r.StatusCode != 429 || r.Header.Get("Retry-After") == "" || !strings.Contains(string(body), "rate_limited") {
		t.Fatalf("expected 429: %d %s", r.StatusCode, body)
	}
	// explanations must not eat the prediction budget
	for i := 0; i < 5; i++ {
		if r, b := e.predict(csrf, "100x", img); r.StatusCode != 200 {
			t.Fatalf("predict %d after explain limit: %d %s", i, r.StatusCode, b)
		}
	}
}

func TestExplainRejectsMalformedModelOutput(t *testing.T) {
	cases := map[string]func(m map[string]any){
		"value out of range": func(m map[string]any) { m["relative"].([][]float64)[2][3] = 1e9 },
		"ragged row":         func(m map[string]any) { m["relative"].([][]float64)[1] = []float64{0, 1} },
		"wrong row count":    func(m map[string]any) { m["relative"] = m["relative"].([][]float64)[:5] },
		"score above 1":      func(m map[string]any) { m["p_oscc"] = 1.5 },
		"bad direction":      func(m map[string]any) { m["check"].(map[string]any)["direction"] = "sideways" },
		"zero cells hidden":  func(m map[string]any) { m["check"].(map[string]any)["cells_hidden"] = 0 },
		"huge grid":          func(m map[string]any) { m["grid"] = []int{500, 500} },
	}
	for name, mut := range cases {
		e := newEnv(t, func(c *Config) { c.Limits.ExplainPerMin, c.Limits.ExplainPerMinIP = 50, 50 })
		e.useInference(t, explainJSON(mut), nil)
		csrf := e.login(t, "doc@hospital.org")
		if r, body := e.explain(csrf, "100x", jpegBytes(t, 256, 256)); r.StatusCode != 503 || strings.Contains(string(body), "relative") {
			t.Errorf("%s: accepted garbage: %d %s", name, r.StatusCode, body)
		}
	}
}

func TestExplainBusyWhenSlotTaken(t *testing.T) {
	old := explainWait
	explainWait = 60 * time.Millisecond
	t.Cleanup(func() { explainWait = old })
	e := newEnv(t, nil)
	e.useInference(t, explainJSON(nil), nil)
	csrf := e.login(t, "doc@hospital.org")
	e.srv.xsem <- struct{}{} // someone else is already computing an explanation
	r, body := e.explain(csrf, "100x", jpegBytes(t, 256, 256))
	if r.StatusCode != 503 || r.Header.Get("Retry-After") == "" || !strings.Contains(string(body), "busy") {
		t.Fatalf("expected busy: %d %s", r.StatusCode, body)
	}
	<-e.srv.xsem
	if r, b := e.explain(csrf, "100x", jpegBytes(t, 256, 256)); r.StatusCode != 200 {
		t.Fatalf("after release: %d %s", r.StatusCode, b)
	}
}

func TestExplainSharesUploadValidation(t *testing.T) {
	e := newEnv(t, func(c *Config) { c.Limits.ExplainPerMin, c.Limits.ExplainPerMinIP = 50, 50 })
	e.useInference(t, explainJSON(nil), nil)
	csrf := e.login(t, "doc@hospital.org")
	if r, _ := e.explain(csrf, "40x", jpegBytes(t, 256, 256)); r.StatusCode != 400 {
		t.Errorf("bad magnification: %d", r.StatusCode)
	}
	if r, _ := e.explain(csrf, "100x", []byte("<html>not an image</html>")); r.StatusCode != 415 {
		t.Errorf("html: %d", r.StatusCode)
	}
	if r, _ := e.explain(csrf, "100x", jpegBytes(t, 64, 64)); r.StatusCode != 422 {
		t.Errorf("tiny: %d", r.StatusCode)
	}
	if r, _ := e.explain(csrf, "100x", nil); r.StatusCode != 400 {
		t.Errorf("no image: %d", r.StatusCode)
	}
	huge := append(jpegBytes(t, 256, 256), make([]byte, 3<<20)...)
	if r, _ := e.explain(csrf, "100x", huge); r.StatusCode != 413 {
		t.Errorf("oversize: %d", r.StatusCode)
	}
}
