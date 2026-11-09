package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"time"
)

type InferResp struct {
	POscc          float64        `json:"p_oscc"`
	Mag            string         `json:"mag"`
	TissueFraction float64        `json:"tissue_fraction"`
	OriginalSize   [2]int         `json:"original_size"`
	QC             []string       `json:"qc"`
	Model          map[string]any `json:"model"`
	LatencyMS      int            `json:"latency_ms"`
}

type InferenceClient struct {
	base, token string
	hc          *http.Client
}

func NewInferenceClient(base, token string) *InferenceClient {
	return &InferenceClient{base: base, token: token, hc: &http.Client{Timeout: 90 * time.Second}}
}

var (
	errInferBadImage = errors.New("image_rejected")
	errInferDown     = errors.New("analysis_unavailable")
	errInferBusy     = errors.New("busy")
)

func (c *InferenceClient) Predict(ctx context.Context, mag string, img []byte) (*InferResp, string, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/v1/predict?mag="+url.QueryEscape(mag), bytes.NewReader(img))
	req.Header.Set("X-Internal-Token", c.token)
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, "", errInferDown
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	switch {
	case resp.StatusCode == http.StatusUnprocessableEntity:
		var e struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(body, &e)
		return nil, e.Error, errInferBadImage
	case resp.StatusCode != http.StatusOK:
		return nil, "", errInferDown
	}
	var r InferResp
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, "", errInferDown
	}
	if math.IsNaN(r.POscc) || r.POscc < 0 || r.POscc > 1 { // never show garbage as a clinical-looking number
		return nil, "", fmt.Errorf("%w: invalid score", errInferDown)
	}
	return &r, "", nil
}

type ExplainCheck struct {
	CellsHidden   int     `json:"cells_hidden"`
	Direction     string  `json:"direction"`
	PBefore       float64 `json:"p_before"`
	PTopHidden    float64 `json:"p_top_hidden"`
	PRandomHidden float64 `json:"p_random_hidden"`
	DeltaTop      float64 `json:"delta_top"`
	DeltaRandom   float64 `json:"delta_random"`
	Supported     bool    `json:"supported"`
}

type ExplainResp struct {
	POscc          float64        `json:"p_oscc"`
	Grid           [2]int         `json:"grid"`
	Window         [2]int         `json:"window"`
	Relative       [][]float64    `json:"relative"`
	MaxAbs         float64        `json:"max_abs"`
	MaxRel         float64        `json:"max_rel"`
	Check          ExplainCheck   `json:"check"`
	Method         string         `json:"method"`
	Passes         int            `json:"passes"`
	TissueFraction float64        `json:"tissue_fraction"`
	Model          map[string]any `json:"model"`
	LatencyMS      int            `json:"latency_ms"`
}

func okProb(p float64) bool { return !math.IsNaN(p) && !math.IsInf(p, 0) && p >= 0 && p <= 1 }
func okNum(v float64) bool  { return !math.IsNaN(v) && !math.IsInf(v, 0) && math.Abs(v) <= 100 }

// validateExplain refuses anything malformed: the UI draws these numbers over a medical image, so garbage must never reach it.
func validateExplain(r *ExplainResp) error {
	rows, cols := r.Grid[0], r.Grid[1]
	if rows < 1 || cols < 1 || rows > 32 || cols > 32 || len(r.Relative) != rows || !okProb(r.POscc) {
		return errors.New("bad grid")
	}
	for _, row := range r.Relative {
		if len(row) != cols {
			return errors.New("ragged grid")
		}
		for _, v := range row {
			if !okNum(v) {
				return errors.New("bad value")
			}
		}
	}
	c := r.Check
	if c.CellsHidden < 1 || c.CellsHidden > rows*cols || (c.Direction != "oscc" && c.Direction != "normal") ||
		!okProb(c.PBefore) || !okProb(c.PTopHidden) || !okProb(c.PRandomHidden) || !okNum(c.DeltaTop) || !okNum(c.DeltaRandom) ||
		!okNum(r.MaxAbs) || !okNum(r.MaxRel) || r.Passes < 1 || r.Passes > 1000 {
		return errors.New("bad check")
	}
	return nil
}

func (c *InferenceClient) Explain(ctx context.Context, mag string, img []byte) (*ExplainResp, string, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/v1/explain?mag="+url.QueryEscape(mag), bytes.NewReader(img))
	req.Header.Set("X-Internal-Token", c.token)
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, "", errInferDown
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	switch {
	case resp.StatusCode == http.StatusUnprocessableEntity:
		var e struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(body, &e)
		return nil, e.Error, errInferBadImage
	case resp.StatusCode == http.StatusServiceUnavailable:
		return nil, "busy", errInferBusy
	case resp.StatusCode != http.StatusOK:
		return nil, "", errInferDown
	}
	var r ExplainResp
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, "", errInferDown
	}
	if err := validateExplain(&r); err != nil {
		return nil, "", fmt.Errorf("%w: %v", errInferDown, err)
	}
	return &r, "", nil
}
