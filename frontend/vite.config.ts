import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Le front (port 5173) relaie /api vers l'API FastAPI (port 8000) : pas de souci de CORS en dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
