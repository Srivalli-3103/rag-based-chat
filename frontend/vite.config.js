import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server runs on 5173 by default. The backend (FastAPI) runs on 8000 —
// different ports count as different "origins" to the browser, which is why
// the backend's CORS middleware (see main.py) is needed for fetch() to work.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
