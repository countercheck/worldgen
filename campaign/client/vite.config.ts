import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  // The fixture world is a 600 KB JSON import. Vite inlines JSON as a module by default,
  // which is fine here and keeps the dev server a single process with no asset serving.
  json: { stringify: false },
});
