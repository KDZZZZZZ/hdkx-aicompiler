#!/usr/bin/env node
/**
 * 同时启动 dev server 与 agent 控制服务。
 *
 * 不引 concurrently 之类的依赖——两个子进程加一段退出处理就够了，
 * 为这点事多一个包不划算。
 */

import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')

const procs = [
  { name: 'control', cmd: process.execPath, args: [join(root, 'server', 'control-server.mjs')] },
  { name: 'vite', cmd: process.execPath, args: [join(root, 'node_modules', 'vite', 'bin', 'vite.js')] },
]

const children = procs.map(({ name, cmd, args }) => {
  const child = spawn(cmd, args, { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] })
  const tag = `[${name}] `
  child.stdout.on('data', (d) => process.stdout.write(tag + d.toString().replace(/\n(?!$)/g, `\n${tag}`)))
  child.stderr.on('data', (d) => process.stderr.write(tag + d.toString().replace(/\n(?!$)/g, `\n${tag}`)))
  child.on('exit', (code) => {
    process.stderr.write(`${tag}退出，code=${code}\n`)
    shutdown(code ?? 0)
  })
  return child
})

let shuttingDown = false
function shutdown(code) {
  if (shuttingDown) return
  shuttingDown = true
  for (const c of children) {
    try {
      c.kill()
    } catch {
      /* 已经退了 */
    }
  }
  process.exit(code)
}

process.on('SIGINT', () => shutdown(0))
process.on('SIGTERM', () => shutdown(0))
