export class ApiError extends Error {
  constructor(status, code, message, retryAfter = 0) {
    super(message);
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

async function request(path, { method = "GET", json, form, csrf } = {}) {
  const headers = {};
  if (json) headers["Content-Type"] = "application/json";
  if (csrf && method !== "GET") headers["X-CSRF-Token"] = csrf;
  let res;
  try {
    res = await fetch(path, {
      method,
      headers,
      body: json ? JSON.stringify(json) : form,
      credentials: "same-origin",
    });
  } catch {
    throw new ApiError(0, "network", "Can't reach the server. Check your connection and try again.");
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const e = data?.error ?? {};
    const retry = e.retryAfterSeconds ?? (Number(res.headers.get("Retry-After")) || 0);
    throw new ApiError(res.status, e.code ?? "error", e.message ?? "Something went wrong.", retry);
  }
  return data;
}

export const api = {
  me: () => request("/api/me"),
  requestCode: (email) => request("/api/auth/request", { method: "POST", json: { email } }),
  verify: (email, code) => request("/api/auth/verify", { method: "POST", json: { email, code } }),
  logout: (csrf) => request("/api/auth/logout", { method: "POST", json: {}, csrf }),
  explain: (csrf, magnification, file) => {
    const form = new FormData();
    form.append("magnification", magnification);
    form.append("image", file);
    return request("/api/explain", { method: "POST", form, csrf });
  },
  predict: (csrf, magnification, file) => {
    const form = new FormData();
    form.append("magnification", magnification);
    form.append("image", file);
    return request("/api/predict", { method: "POST", form, csrf });
  },
};
