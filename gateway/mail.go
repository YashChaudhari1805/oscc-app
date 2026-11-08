package main

import (
	"fmt"
	"log/slog"
	"net/smtp"
	"strings"
	"time"
)

type Mailer interface{ SendCode(to, code string) error }

type smtpMailer struct{ cfg *Config }

func (m smtpMailer) SendCode(to, code string) error {
	body := "From: " + m.cfg.SMTPFrom + "\r\nTo: " + to + "\r\nSubject: Your OSCC Assistant sign-in code\r\n" +
		"MIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n" +
		fmt.Sprintf("Your sign-in code is %s\r\n\r\nIt expires in 10 minutes. If you did not request it, ignore this email.\r\n", code)
	var auth smtp.Auth
	if m.cfg.SMTPUser != "" {
		auth = smtp.PlainAuth("", m.cfg.SMTPUser, m.cfg.SMTPPass, m.cfg.SMTPHost) // refuses to send creds over plaintext
	}
	done := make(chan error, 1)
	go func() {
		done <- smtp.SendMail(m.cfg.SMTPHost+":"+m.cfg.SMTPPort, auth, extractAddr(m.cfg.SMTPFrom), []string{to}, []byte(body))
	}()
	select {
	case err := <-done:
		return err
	case <-time.After(20 * time.Second):
		return fmt.Errorf("smtp timeout")
	}
}

func extractAddr(from string) string {
	if i := strings.Index(from, "<"); i >= 0 {
		if j := strings.Index(from[i:], ">"); j > 0 {
			return from[i+1 : i+j]
		}
	}
	return from
}

// devMailer prints the code to the server log. Refused in production by LoadConfig.
type devMailer struct{}

func (devMailer) SendCode(to, code string) error {
	slog.Warn("DEV_ECHO_OTP sign-in code", "to", to, "code", code)
	return nil
}
