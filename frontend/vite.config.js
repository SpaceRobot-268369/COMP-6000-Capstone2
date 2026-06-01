import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Vite 5.4+ blocks requests whose Host header is a domain not in this list
    // (DNS-rebinding guard); raw IPs and localhost are always allowed. The
    // public deployment is reached via the adelaideuni.cloud hostname, so allow
    // that zone explicitly — otherwise the domain 403s while the bare IP works.
    allowedHosts: [".adelaideuni.cloud"],
    watch: { usePolling: true },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:4000",
        changeOrigin: true,
      },
    },
  }
});
