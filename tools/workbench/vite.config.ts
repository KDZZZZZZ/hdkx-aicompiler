import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 本地优先：dev server 不做任何外部代理，bundle 全部在浏览器内解析。
// 不配 '@' 别名——源码一律用相对路径导入，省掉一份需要和 tsconfig 保持同步的配置，
// 也避免为了 fileURLToPath 而引入 @types/node。
export default defineConfig({
  plugins: [react()],
  worker: {
    format: 'es',
  },
  server: {
    port: 5273,
    // fixtures 目录通过 publicDir 暴露，方便无文件选择器时直接加载样例 bundle。
    fs: { allow: ['..'] },
  },
  publicDir: 'fixtures',
  build: {
    target: 'es2022',
    sourcemap: true,
  },
})
