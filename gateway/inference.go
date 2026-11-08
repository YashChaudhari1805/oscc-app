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
