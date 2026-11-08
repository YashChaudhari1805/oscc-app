import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api.js";

/* ---------- helpers ---------- */

const MAX_MB = 20;
const sniff = async (file) => {
  const b = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "jpeg";
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return "png";
  return null;
};
const loadDims = (url) =>
  new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res({ w: im.naturalWidth, h: im.naturalHeight });
    im.onerror = rej;
    im.src = url;
  });
const fmtBytes = (n) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

function useCountdown(initial) {
  const [s, setS] = useState(initial);
  useEffect(() => {
    if (s <= 0) return;
    const t = setTimeout(() => setS((x) => x - 1), 1000);
    return () => clearTimeout(t);
  }, [s]);
  return [s, setS];
}

function useTheme() {
  const [theme, setTheme] = useState(() => (document.documentElement.dataset.theme === "dark" ? "dark" : "light"));
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("oscc-theme", next); } catch { /* storage blocked: still works for this visit */ }
    setTheme(next);
  };
  return [theme, toggle];
}

const Icon = ({ d, size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="currentColor">
    <path d={d} />
  </svg>
);
const ICONS = {
  upload: "M11 16V7.8l-3.1 3.1L6.5 9.5 12 4l5.5 5.5-1.4 1.4L13 7.8V16h-2zM5 19h14v2H5v-2z",
  logout: "m17 7-1.4 1.4L18.2 11H9v2h9.2l-2.6 2.6L17 17l5-5-5-5zM4 5h8V3H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h8v-2H4V5z",
  warn: "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z",
  print: "M19 8H5a3 3 0 0 0-3 3v6h4v4h12v-4h4v-6a3 3 0 0 0-3-3zm-3 11H8v-5h8v5zm3-7a1 1 0 1 1 0-2 1 1 0 0 1 0 2zM18 3H6v4h12V3z",
  close: "M19 6.4 17.6 5 12 10.6 6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12z",
  check: "M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z",
  sun: "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM11 1h2v3h-2V1zm0 19h2v3h-2v-3zM1 11h3v2H1v-2zm19 0h3v2h-3v-2zM4.2 5.6l1.4-1.4 2.1 2.1-1.4 1.4-2.1-2.1zm12.1 12.1 1.4-1.4 2.1 2.1-1.4 1.4-2.1-2.1zM5.6 19.8l-1.4-1.4 2.1-2.1 1.4 1.4-2.1 2.1zM18.4 4.2l1.4 1.4-2.1 2.1-1.4-1.4 2.1-2.1z",
  moon: "M12.1 22A10 10 0 0 1 9.6 2.3a.8.8 0 0 1 .9 1.1A8 8 0 0 0 20.6 13.5a.8.8 0 0 1 1.1.9A10 10 0 0 1 12.1 22z",
};

/* ---------- the graticule: ribbed arcs used as the visual signature ---------- */

function Arc() {
  const ring = (n, r1, r2, cls) =>
    Array.from({ length: n }, (_, i) => {
      const a = Math.PI + (Math.PI / 2) * (i / (n - 1));
      return (
        <line key={cls + i} className={`rib ${cls}`} style={{ "--i": i }}
          x1={600 + r1 * Math.cos(a)} y1={600 + r1 * Math.sin(a)} x2={600 + r2 * Math.cos(a)} y2={600 + r2 * Math.sin(a)} />
      );
    });
  return (
    <svg className="arc" viewBox="0 0 600 600" aria-hidden="true" focusable="false">
      <circle className="arc-disc" cx="600" cy="600" r="400" />
      {ring(46, 412, 520, "outer")}
      {ring(40, 296, 372, "inner")}
    </svg>
  );
}

/* Score bar: a plain linear scale with the uncertain band shaded and a marker at the model score. */
function ScoreBar({ p, band, tone }) {
  const pct = Math.round(p * 100);
  const lo = Math.round(band.low * 100), hi = Math.round(band.high * 100);
  return (
    <div className={`scorebar tone-${tone}`} role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="Model score for OSCC">
      <div className="score-read">
        <span className="score-num">{pct}<small>%</small></span>
        <span className="score-cap">model score for OSCC</span>
      </div>
      <div className="track" aria-hidden="true">
        <span className="zone z-low" style={{ width: `${lo}%` }} />
        <span className="zone z-mid" style={{ width: `${hi - lo}%` }} />
        <span className="zone z-high" style={{ width: `${100 - hi}%` }} />
        <span className="marker" style={{ left: `${pct}%` }} />
      </div>
      <div className="scale" aria-hidden="true">
        <span style={{ left: 0 }}>0</span>
        <span style={{ left: `${lo}%` }}>{lo}</span>
        <span style={{ left: `${hi}%` }}>{hi}</span>
        <span style={{ left: "100%" }}>100</span>
      </div>
      <div className="legend" aria-hidden="true">
        <span><i className="z-low" />Normal-like</span>
        <span><i className="z-mid" />Uncertain</span>
        <span><i className="z-high" />OSCC-like</span>
      </div>
    </div>
  );
}

/* ---------- app frame ---------- */

function AppBar({ user, onSignOut, theme, onTheme }) {
  return (
    <header className="appbar no-print">
      <div className="brand">
        <span className="brand-name">OSCC Assistant</span>
      </div>
      <div className="appbar-actions">
        {user && <span className="user-chip" title={user.email}>{user.email}</span>}
        <button className="icon-btn" onClick={onTheme} aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Light mode" : "Dark mode"}>
          <Icon d={theme === "dark" ? ICONS.sun : ICONS.moon} />
        </button>
        {user && (
          <button className="icon-btn" onClick={onSignOut} aria-label="Sign out" title="Sign out">
            <Icon d={ICONS.logout} />
          </button>
        )}
      </div>
    </header>
  );
}

const Pill = () => (
  <span className="pill"><i aria-hidden="true" />Research prototype</span>
);

/* ---------- sign in ---------- */

function SignIn({ onDone, notice }) {
  const [step, setStep] = useState("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [wait, setWait] = useCountdown(0);
  const codeRef = useRef(null);

  const sendCode = async (e) => {
    e?.preventDefault();
    if (busy || wait > 0) return;
    setBusy(true);
    setError("");
    try {
      await api.requestCode(email.trim());
      setStep("code");
      setCode("");
      setWait(30);
      setTimeout(() => codeRef.current?.focus(), 50);
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) setWait(err.retryAfter || 60);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const verify = useCallback(
    async (c) => {
      setBusy(true);
      setError("");
      try {
        onDone(await api.verify(email.trim(), c));
      } catch (err) {
        if (err instanceof ApiError && err.status === 429) setWait(err.retryAfter || 60);
        setError(err.message);
        setCode("");
        codeRef.current?.focus();
      } finally {
        setBusy(false);
      }
    },
    [email, onDone, setWait]
  );

  const onCode = (v) => {
    const digits = v.replace(/\D/g, "").slice(0, 6);
    setCode(digits);
    if (digits.length === 6 && !busy) verify(digits);
  };

  return (
    <section className="signin" aria-labelledby="signin-title">
      <h2 id="signin-title">Sign in</h2>
      {notice && <p className="note warn" role="status"><Icon d={ICONS.warn} size={18} />{notice}</p>}
      {step === "email" ? (
        <form className="stack" onSubmit={sendCode} noValidate>
          <p className="soft">Enter your work e-mail. If it is authorised, we will send a 6-digit code.</p>
          <label className="field">
            <span>E-mail</span>
            <input type="email" autoComplete="email" inputMode="email" autoFocus required value={email}
              onChange={(e) => setEmail(e.target.value)} aria-invalid={!!error} aria-describedby="signin-err" />
          </label>
          <p id="signin-err" className="field-error" role="alert">{error}</p>
          <button className="btn solid block" type="submit" disabled={busy || !email.includes("@") || wait > 0}>
            {busy ? "Sending…" : wait > 0 ? `Try again in ${wait}s` : "Send code"}
          </button>
        </form>
      ) : (
        <div className="stack">
          <p className="soft">
            If <strong>{email}</strong> is authorised, a code is on its way. It expires in 10 minutes.
          </p>
          <label className="field">
            <span>6-digit code</span>
            <input ref={codeRef} className="code" inputMode="numeric" autoComplete="one-time-code" maxLength={6}
              value={code} onChange={(e) => onCode(e.target.value)} disabled={busy}
              aria-invalid={!!error} aria-describedby="code-err" placeholder="······" />
          </label>
          <p id="code-err" className="field-error" role="alert">{error}</p>
          <div className="row between">
            <button className="btn quiet" onClick={() => { setStep("email"); setError(""); }}>Change e-mail</button>
            <button className="btn quiet" onClick={sendCode} disabled={busy || wait > 0}>
              {wait > 0 ? `Resend in ${wait}s` : "Resend code"}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}

function Landing({ onDone, notice }) {
  return (
    <main className="stage hero">
      <Arc />
      <div className="hero-copy">
        <Pill />
        <h1 className="display">A second look at <em>every field.</em></h1>
        <p className="lede">
          Upload a microscope image of oral tissue and get a model score for OSCC-like patterns.
          It supports a pathologist's review and never replaces it.
        </p>
        <details className="limits">
          <summary className="btn glass">Before you use it</summary>
          <ul>
            <li>Trained on one public dataset of oral histology. It has not been tested on images from other labs or scanners.</li>
            <li>It answers one question: does this field look more like normal epithelium or like OSCC? It does not grade tumours.</li>
            <li>Each magnification has its own model, so choose the one your image was captured at.</li>
            <li>A low score does not rule out disease.</li>
          </ul>
        </details>
      </div>
      <SignIn onDone={onDone} notice={notice} />
    </main>
  );
}

/* ---------- analyse ---------- */

const LABELS = {
  oscc_like: { title: "Features suggestive of OSCC", tone: "error", hint: "A pathologist should confirm this." },
  normal_like: { title: "Features consistent with normal epithelium", tone: "ok", hint: "A low score does not rule out disease." },
  uncertain: { title: "Uncertain result", tone: "warn", hint: "The model is not confident. Expert review is needed." },
};

function Segmented({ value, onChange, options, label }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o} type="button" role="radio" aria-checked={value === o}
          className={value === o ? "seg on" : "seg"} onClick={() => onChange(o)}>
          {value === o && <Icon d={ICONS.check} size={16} />}
          {o}
        </button>
      ))}
    </div>
  );
}

function Dropzone({ onFile, disabled }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);
  return (
    <div
      className={`dropzone${over ? " over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (!disabled && e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); }}
    >
      <span className="drop-icon"><Icon d={ICONS.upload} size={26} /></span>
      <p className="drop-title">Drop a microscope image here</p>
      <p className="soft small">JPEG or PNG, up to {MAX_MB} MB</p>
      <button className="btn outline" type="button" disabled={disabled} onClick={() => input.current?.click()}>
        Choose image
      </button>
      <input ref={input} type="file" accept="image/jpeg,image/png" hidden data-testid="file-input"
        onChange={(e) => { if (e.target.files[0]) onFile(e.target.files[0]); e.target.value = ""; }} />
    </div>
  );
}

function ResultPanel({ r, email, onReset }) {
  const L = LABELS[r.label];
  const d = new Date(r.createdAt);
  const when = d.toLocaleString();
  const pct = Math.round(r.pOscc * 100);
  const lo = Math.round(r.band.low * 100), hi = Math.round(r.band.high * 100);
  const img = r.imageInfo;
  return (
    <section className="panel result" aria-live="polite" aria-label="Result">
      <div className={`verdict tone-box-${L.tone}`}>
        <h2>{L.title}</h2>
        <p>{L.hint}</p>
      </div>
      <ScoreBar p={r.pOscc} band={r.band} tone={L.tone} />
      <p className="soft small center">Scores between {lo}% and {hi}% are treated as uncertain.</p>
      {r.qc.map((q) => (
        <p key={q.code} className="note warn"><Icon d={ICONS.warn} size={18} />{q.message}</p>
      ))}
      <dl className="facts">
        <div><dt>Magnification</dt><dd>{r.magnification}</dd></div>
        <div><dt>Tissue in field</dt><dd>{Math.round(r.tissueFraction * 100)}%</dd></div>
        <div><dt>Model</dt><dd>{r.model?.version}</dd></div>
        <div><dt>Analysed</dt><dd>{when}</dd></div>
      </dl>
      <p className="soft small">{r.notice}</p>
      <div className="row between no-print">
        <button className="btn quiet" onClick={() => window.print()}><Icon d={ICONS.print} size={18} />Print report</button>
        <button className="btn solid" onClick={onReset}>Analyse another image</button>
      </div>

      {/* ---- printable report (hidden on screen) ---- */}
      <article className="print-only report">
        <header className="rp-head">
          <div>
            <p className="rp-kicker">Research output</p>
            <h1>OSCC Assistant: Image Analysis Report</h1>
          </div>
          <dl className="rp-meta">
            <div><dt>Report no.</dt><dd>{r.resultId}</dd></div>
            <div><dt>Date</dt><dd>{when}</dd></div>
            {email && <div><dt>Run by</dt><dd>{email}</dd></div>}
          </dl>
        </header>

        <p className="rp-banner">Research prototype. Not a diagnosis, not a medical device, and not validated for clinical use.</p>

        <section className="rp-sec">
          <h2>1. Case details</h2>
          <table className="rp-form"><tbody>
            <tr><th>Case / patient ID</th><td /><th>Specimen site</th><td /></tr>
            <tr><th>Requesting clinician</th><td /><th>Slide / block no.</th><td /></tr>
          </tbody></table>
          <p className="rp-note">Fill in by hand if needed. The app does not store any patient information.</p>
        </section>

        <section className="rp-sec">
          <h2>2. Summary of result</h2>
          <div className={`rp-verdict tone-box-${L.tone}`}>
            <strong>{L.title}</strong>
            <span>{L.hint}</span>
          </div>
          <ScoreBar p={r.pOscc} band={r.band} tone={L.tone} />
          <p className="rp-note">
            The score is the model's calibrated probability that this field resembles OSCC rather than normal epithelium.
            Scores from {lo}% to {hi}% are reported as uncertain.
          </p>
        </section>

        <section className="rp-sec rp-keep">
          <h2>3. Analysed image</h2>
          <div className="rp-image">
            {r.previewUrl && <img src={r.previewUrl} alt="Analysed field" />}
            <table className="rp-table"><tbody>
              <tr><th>Magnification</th><td>{r.magnification}</td></tr>
              {img && <tr><th>Dimensions</th><td>{img.w} × {img.h} px</td></tr>}
              {img && <tr><th>File size</th><td>{fmtBytes(img.size)}</td></tr>}
              <tr><th>Tissue in field</th><td>{Math.round(r.tissueFraction * 100)}%</td></tr>
            </tbody></table>
          </div>
        </section>

        <section className="rp-sec rp-keep">
          <h2>4. Image quality checks</h2>
          {r.qc.length ? (
            <ul className="rp-list">{r.qc.map((q) => <li key={q.code}>{q.message}</li>)}</ul>
          ) : (
            <p>No quality warnings were raised for this image.</p>
          )}
        </section>

        <section className="rp-sec rp-keep">
          <h2>5. Method</h2>
          <table className="rp-table"><tbody>
            <tr><th>Task</th><td>Binary classification: normal epithelium vs oral squamous cell carcinoma (OSCC)</td></tr>
            <tr><th>Model</th><td>{r.model?.version ?? "n/a"} ({r.magnification} model)</td></tr>
            <tr><th>Preprocessing</th><td>Area-interpolated resize and Macenko stain normalisation, identical to training</td></tr>
            <tr><th>Output</th><td>Calibrated probability of OSCC, shown as a percentage</td></tr>
            <tr><th>Training data</th><td>One public oral histology dataset; patient-level cross-validation; no external validation</td></tr>
          </tbody></table>
        </section>

        <section className="rp-sec rp-keep">
          <h2>6. Limitations</h2>
          <ul className="rp-list">
            <li>The model has not been tested on images from other laboratories, stains or scanners.</li>
            <li>It looks at a single microscope field. It does not assess the whole slide and does not grade tumours.</li>
            <li>A low score does not rule out disease, and a high score does not confirm it.</li>
            <li>Results must be reviewed by a qualified pathologist and must not be used on their own for any clinical decision.</li>
          </ul>
        </section>

        <section className="rp-sec rp-keep rp-sign">
          <h2>7. Review</h2>
          <div className="rp-sign-grid">
            <div><span>Reviewer name</span></div>
            <div><span>Signature</span></div>
            <div><span>Date</span></div>
          </div>
          <p className="rp-note">Comments:</p>
          <div className="rp-lines"><i /><i /></div>
        </section>

        <footer className="rp-foot">
          <span>OSCC Assistant · Report {r.resultId}</span>
          <span>Research prototype. Not a medical device.</span>
        </footer>
      </article>
    </section>
  );
}

function Workspace({ csrf, email, onExpired }) {
  const [mag, setMag] = useState("100x");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [wait, setWait] = useCountdown(0);

  useEffect(() => () => preview && URL.revokeObjectURL(preview.url), [preview]);

  const pick = async (f) => {
    setError("");
    if (f.size > MAX_MB * 1048576) return setError(`That file is larger than ${MAX_MB} MB.`);
    if (!(await sniff(f))) return setError("Only JPEG and PNG images are supported.");
    const url = URL.createObjectURL(f);
    try {
      const d = await loadDims(url);
      setResult(null);
      setFile(f);
      setPreview({ url, ...d, size: f.size });
    } catch {
      URL.revokeObjectURL(url);
      setError("This file could not be read as an image.");
    }
  };
  const clear = () => { setResult(null); setFile(null); setPreview(null); setError(""); };

  const run = async () => {
    if (!file || busy || wait > 0) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.predict(csrf, mag, file);
      setResult({ ...r, previewUrl: preview?.url, imageInfo: preview ? { w: preview.w, h: preview.h, size: preview.size } : null });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) return onExpired();
      if (err instanceof ApiError && err.status === 429) setWait(err.retryAfter || 30);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const small = preview && Math.min(preview.w, preview.h) < 480;

  return (
    <>
      <section className="stage band no-print">
        <Arc />
        <div className="hero-copy">
          <Pill />
          <h1 className="display sm">Analyse <em>a field.</em></h1>
          <p className="lede">Choose the magnification, add one image, and run the model.</p>
        </div>
      </section>

      <main className="workspace">
        <section className="panel input no-print" aria-labelledby="in-title">
          <h2 id="in-title">Image</h2>
          <div className="row between wrap">
            <span className="label">Magnification</span>
            <Segmented value={mag} onChange={setMag} options={["100x", "400x"]} label="Magnification of the image" />
          </div>
          <p className="soft small">Choose the objective the image was captured at. Each magnification uses its own model.</p>

          {!preview ? (
            <Dropzone onFile={pick} disabled={busy} />
          ) : (
            <div className="preview">
              <img src={preview.url} alt="Selected field preview" />
              <div className="preview-meta">
                <span>{preview.w}×{preview.h} px, {fmtBytes(preview.size)}</span>
                <button className="icon-btn" onClick={clear} aria-label="Remove image" disabled={busy}><Icon d={ICONS.close} /></button>
              </div>
            </div>
          )}
          {small && <p className="note warn"><Icon d={ICONS.warn} size={18} />This image is small. The model was trained on fields of about 1600×1200 px, so accuracy may be lower.</p>}
          <p className="field-error" role="alert">{error}</p>
          {busy && <div className="progress" role="progressbar" aria-label="Analysing"><div /></div>}
          <button className="btn solid block" onClick={run} disabled={!file || busy || wait > 0}>
            {busy ? "Analysing…" : wait > 0 ? `Please wait ${wait}s` : "Analyse"}
          </button>
          <p className="soft small">Do not upload images that show patient names or IDs. Metadata is removed, but text burned into the picture is not.</p>
        </section>

        {result ? (
          <ResultPanel r={result} email={email} onReset={clear} />
        ) : (
          <section className="panel empty no-print" aria-label="Result">
            <h2>Your result appears here</h2>
            <p className="soft">After you run the model you will get:</p>
            <ul className="empty-list">
              <li><span>1</span>A model score from 0 to 100%</li>
              <li><span>2</span>Quality warnings for the image</li>
              <li><span>3</span>A report you can print or save as PDF</li>
            </ul>
          </section>
        )}
      </main>
    </>
  );
}

/* ---------- root ---------- */

export default function App() {
  const [boot, setBoot] = useState(true);
  const [user, setUser] = useState(null);
  const [notice, setNotice] = useState("");
  const [theme, toggleTheme] = useTheme();

  useEffect(() => {
    api.me().then(setUser).catch(() => {}).finally(() => setBoot(false));
  }, []);

  const signOut = async () => {
    try { await api.logout(user.csrfToken); } catch { /* cookie may already be gone */ }
    setUser(null);
    setNotice("");
  };

  return (
    <div className="shell">
      <div className="frame">
        <AppBar user={user} onSignOut={signOut} theme={theme} onTheme={toggleTheme} />
        {boot ? (
          <div className="stage boot"><div className="progress" role="progressbar" aria-label="Loading"><div /></div></div>
        ) : user ? (
          <Workspace csrf={user.csrfToken} email={user.email} onExpired={() => { setUser(null); setNotice("Your session expired. Please sign in again."); }} />
        ) : (
          <Landing onDone={(u) => { setNotice(""); setUser(u); }} notice={notice} />
        )}
      </div>
      <footer className="legal no-print">
        <p>Research prototype. Not a medical device. Not validated for clinical use.</p>
        <p>Images are analysed in memory and are not stored.</p>
      </footer>
    </div>
  );
}
