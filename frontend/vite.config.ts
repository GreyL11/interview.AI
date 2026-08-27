import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Tauri drives the dev server, so the port is fixed and failures must be loud
// rather than silently landing on another port the shell isn't pointed at.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      // The Rust side and the Python backend have their own reload paths.
      ignored: ["**/src-tauri/**", "**/backend/**"],
    },
  },
  build: {
    // WebView2 on Windows tracks Chromium, so no legacy transpilation needed.
    target: "chrome110",
    sourcemap: true,
    outDir: "dist",
  },
});
