import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
    dedupe: ['vue'],
  },
  optimizeDeps: {
    // Linked file: packages — exclude so Vite loads fresh dist (avoids stale icon exports).
    include: ['reka-ui'],
    exclude: ['@danqing/dq-ui', '@danqing/dq-shell'],
  },
  server: {
    host: '127.0.0.1',
    port: Number(process.env.DQ_FRONTEND_PORT || 5800),
    strictPort: true,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${process.env.DQ_BACKEND_PORT || 7800}`,
        changeOrigin: true,
      },
      // OpenAI-compatible LLM/VLM routes (not under /api)
      '/v1': {
        target: `http://127.0.0.1:${process.env.DQ_BACKEND_PORT || 7800}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: resolve(__dirname, '../out/frontend/dist'),
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      onwarn(warning, warn) {
        // @vueuse/core (via dq-ui): harmless PURE annotation placement noise
        if (warning.message?.includes('__PURE__')) return;
        warn(warning);
      },
    },
  },
});