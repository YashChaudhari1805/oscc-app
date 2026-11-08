package main

import (
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Limits struct {
	GlobalPerMinIP                     int
	OTPReqPerIPHour, OTPReqPerEmail15m int
	VerifyPerIP15m, VerifyPerEmail15m  int
	PredictPerMin, PredictPerHour      int
	PredictPerDay, PredictPerMinIP     int
}

type Config struct {
	Env           string
	Addr          string
	InferenceURL  string
	InternalToken string
	SessionSecret []byte

	AllowedEmails  map[string]bool
	AllowedDomains map[string]bool
	AllowAnyEmail  bool // development only

	CookieSecure bool
	CookieName   string
	TrustedProxy bool

	SMTPHost, SMTPPort, SMTPUser, SMTPPass, SMTPFrom string
	DevEchoOTP                                       bool

	WebDir    string
	AuditPath string

	MaxUploadBytes     int64
	MaxConcurrentInfer int
	LowBand, HighBand  float64
	Limits             Limits
}

func envStr(k, d string) string {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		return v
	}
	return d
}
func envInt(k string, d int) int {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return d
}
func envFloat(k string, d float64) float64 {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return d
}
func envBool(k string, d bool) bool {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		b, err := strconv.ParseBool(v)
		if err == nil {
			return b
		}
	}
	return d
}
func splitSet(s string) map[string]bool {
	m := map[string]bool{}
	for _, p := range strings.Split(s, ",") {
		if p = strings.ToLower(strings.TrimSpace(p)); p != "" {
			m[p] = true
		}
	}
	return m
}

func LoadConfig() (*Config, error) {
	prod := envStr("APP_ENV", "development") == "production"
	c := &Config{
		Env:                envStr("APP_ENV", "development"),
		Addr:               envStr("GATEWAY_ADDR", "127.0.0.1:8080"),
		InferenceURL:       strings.TrimRight(envStr("INFERENCE_URL", "http://127.0.0.1:9000"), "/"),
		InternalToken:      os.Getenv("INTERNAL_TOKEN"),
		AllowedEmails:      splitSet(os.Getenv("ALLOWED_EMAILS")),
		AllowedDomains:     splitSet(os.Getenv("ALLOWED_DOMAINS")),
		AllowAnyEmail:      envBool("ALLOW_ANY_EMAIL", false),
		CookieSecure:       envBool("COOKIE_SECURE", prod),
		TrustedProxy:       envBool("TRUSTED_PROXY", false),
		SMTPHost:           os.Getenv("SMTP_HOST"),
		SMTPPort:           envStr("SMTP_PORT", "587"),
		SMTPUser:           os.Getenv("SMTP_USER"),
		SMTPPass:           os.Getenv("SMTP_PASS"),
		SMTPFrom:           os.Getenv("SMTP_FROM"),
		DevEchoOTP:         envBool("DEV_ECHO_OTP", false),
		WebDir:             envStr("WEB_DIR", "../web/dist"),
		AuditPath:          envStr("AUDIT_LOG", "audit.jsonl"),
		MaxUploadBytes:     int64(envInt("MAX_UPLOAD_MB", 20)) << 20,
		MaxConcurrentInfer: envInt("MAX_CONCURRENT_INFER", 4),
		LowBand:            envFloat("BAND_LOW", 0.20),
		HighBand:           envFloat("BAND_HIGH", 0.80),
		Limits: Limits{
			GlobalPerMinIP:  envInt("LIMIT_GLOBAL_PER_MIN_IP", 120),
			OTPReqPerIPHour: envInt("LIMIT_OTP_REQ_PER_IP_HOUR", 10), OTPReqPerEmail15m: envInt("LIMIT_OTP_REQ_PER_EMAIL_15M", 3),
			VerifyPerIP15m: envInt("LIMIT_VERIFY_PER_IP_15M", 20), VerifyPerEmail15m: envInt("LIMIT_VERIFY_PER_EMAIL_15M", 10),
			PredictPerMin: envInt("LIMIT_PREDICT_PER_MIN", 5), PredictPerHour: envInt("LIMIT_PREDICT_PER_HOUR", 40),
			PredictPerDay: envInt("LIMIT_PREDICT_PER_DAY", 150), PredictPerMinIP: envInt("LIMIT_PREDICT_PER_MIN_IP", 15),
		},
	}
	c.CookieName = "oscc_session"
	if c.CookieSecure {
		c.CookieName = "__Host-oscc_session" // browser-enforced: Secure, Path=/, no Domain
	}
	if s := os.Getenv("SESSION_SECRET"); s != "" {
		c.SessionSecret = []byte(s)
	}
	if len(c.InternalToken) < 16 {
		return nil, errors.New("INTERNAL_TOKEN must be set (>=16 chars) and match the inference service")
	}
	if len(c.SessionSecret) == 0 && !prod {
		c.SessionSecret = make([]byte, 32)
		_, _ = rand.Read(c.SessionSecret)
	}
	if len(c.SessionSecret) < 32 {
		return nil, errors.New("SESSION_SECRET must be set (>=32 chars)")
	}
	if c.LowBand >= c.HighBand || c.LowBand < 0 || c.HighBand > 1 {
		return nil, errors.New("BAND_LOW must be < BAND_HIGH within [0,1]")
	}
	if prod { // fail closed: refuse to start in an insecure production configuration
		switch {
		case c.AllowAnyEmail:
			return nil, errors.New("ALLOW_ANY_EMAIL is not permitted in production")
		case c.DevEchoOTP:
			return nil, errors.New("DEV_ECHO_OTP is not permitted in production")
		case !c.CookieSecure:
			return nil, errors.New("COOKIE_SECURE must be true in production (serve over HTTPS)")
		case c.SMTPHost == "" || c.SMTPFrom == "":
			return nil, errors.New("SMTP_HOST and SMTP_FROM are required in production")
		case len(c.AllowedEmails) == 0 && len(c.AllowedDomains) == 0:
			return nil, errors.New("set ALLOWED_EMAILS and/or ALLOWED_DOMAINS in production")
		}
	} else if !c.AllowAnyEmail && len(c.AllowedEmails) == 0 && len(c.AllowedDomains) == 0 {
		return nil, fmt.Errorf("no one is allowed to sign in: set ALLOWED_EMAILS / ALLOWED_DOMAINS (or ALLOW_ANY_EMAIL=true for local dev)")
	}
	if c.SMTPHost == "" && !c.DevEchoOTP {
		return nil, errors.New("no way to deliver sign-in codes: set SMTP_HOST or (dev only) DEV_ECHO_OTP=true")
	}
	return c, nil
}
