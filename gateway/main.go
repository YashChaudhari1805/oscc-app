package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	cfg, err := LoadConfig()
	if err != nil {
		log.Error("configuration error", "err", err)
		os.Exit(1)
	}
	audit, err := NewAudit(cfg.AuditPath)
	if err != nil {
		log.Error("cannot open audit log", "err", err)
		os.Exit(1)
	}
	var mailer Mailer = devMailer{}
	if cfg.SMTPHost != "" {
		mailer = smtpMailer{cfg}
	}
	srv := NewServer(cfg, mailer, audit, log)
	go func() {
		for range time.Tick(time.Minute) {
			srv.Sweep()
		}
	}()
	hs := &http.Server{
		Addr: cfg.Addr, Handler: srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 60 * time.Second,
		WriteTimeout: 120 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 << 10,
	}
	go func() {
		log.Info("gateway listening", "addr", cfg.Addr, "env", cfg.Env)
		if err := hs.ListenAndServe(); err != http.ErrServerClosed {
			log.Error("server error", "err", err)
			os.Exit(1)
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = hs.Shutdown(ctx)
}
