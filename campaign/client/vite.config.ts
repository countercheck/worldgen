import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The client and the server are one application split over two ports in development.
    // Proxying rather than pointing the client at http://localhost:3000 keeps everything
    // same-origin, which is what makes the join cookie and the WebSocket work without a
    // CORS story that would exist only in development.
    proxy: {
      '/api': { target: 'http://127.0.0.1:3000', changeOrigin: true, ws: true },
      '/j': { target: 'http://127.0.0.1:3000', changeOrigin: true },
    },
  },
  // The fixture world is a 600 KB JSON import. Vite inlines JSON as a module by default,
  // which is fine here and keeps the dev server a single process with no asset serving.
  json: { stringify: false },
});
