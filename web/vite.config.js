import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` proxies /api to the Go gateway so cookies/CSRF behave like production.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8080" } },
  build: { sourcemap: false, target: "es2020", assetsInlineLimit: 0 }  // never inline: the CSP forbids data: fonts,
});
