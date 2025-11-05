// vite.config.js
import { resolve } from 'path'

export default {
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/pages/sim.html'),
        monitor: resolve(__dirname, 'src/pages/monitor.html'),
        real: resolve(__dirname, 'src/pages/real.html')
      }
    }
  },
  server: {
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
};