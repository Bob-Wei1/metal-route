import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `vite dev` the SPA runs on its own port; proxy the API calls to the
// running `mr-server` (default port 1234) so `/api/*` works without CORS hassle.
// `npm run build` emits to `web/dist`, which `mr-server` serves as its fallback.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://localhost:1234",
    },
  },
});
