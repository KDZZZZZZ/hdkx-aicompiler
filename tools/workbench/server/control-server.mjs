#!/usr/bin/env node
/**
 * 桌布控制服务：让 CLI（以及背后的 agent）驱动正在浏览的那个页面。
 *
 * 为什么是 SSE + POST 而不是 WebSocket：
 * Node 没有内置 WebSocket 服务端，用 ws 包就要为一件小事引入依赖，
 * 手写握手又纯属浪费。SSE 是浏览器内置的（EventSource），下行足够；
 * 上行用普通 POST 就行。零依赖、零构建，行为还更容易调试（curl 就能测）。
 *
 * 拓扑：
 *   CLI  --POST /command-->  server  --SSE /events-->  浏览器
 *   CLI  <--GET /state----   server  <--POST /state--  浏览器
 *
 * 状态是浏览器推上来的快照，服务端只做中转与缓存，自己不持有任何真相。
 * 这样 agent 读到的永远是页面真实状态，而不是命令发出后的假设状态。
 */

import { createServer } from 'node:http'

const PORT = Number(process.env.KXC_WB_PORT ?? 5274)

/** 已连接的浏览器（SSE 响应流）。允许多开，命令会广播给所有页面。 */
const clients = new Set()

/** 浏览器最近一次推上来的状态快照。 */
let latestState = null
let latestStateAt = 0

/** 等待某条命令回执的 CLI 请求：commandId -> {resolve, timer} */
const pendingAcks = new Map()
let commandSeq = 0

function json(res, code, body) {
  const text = JSON.stringify(body)
  res.writeHead(code, {
    'content-type': 'application/json; charset=utf-8',
    'access-control-allow-origin': '*',
    'access-control-allow-headers': 'content-type',
    'content-length': Buffer.byteLength(text),
  })
  res.end(text)
}

async function readBody(req) {
  const chunks = []
  for await (const c of req) chunks.push(c)
  if (chunks.length === 0) return null
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'))
  } catch {
    return null
  }
}

function broadcast(event, payload) {
  const frame = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`
  for (const res of clients) {
    try {
      res.write(frame)
    } catch {
      clients.delete(res)
    }
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://localhost:${PORT}`)

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'access-control-allow-origin': '*',
      'access-control-allow-headers': 'content-type',
      'access-control-allow-methods': 'GET,POST,OPTIONS',
    })
    return res.end()
  }

  // --- 浏览器订阅命令流 ---
  if (url.pathname === '/events' && req.method === 'GET') {
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'access-control-allow-origin': '*',
    })
    res.write(`event: hello\ndata: ${JSON.stringify({ port: PORT })}\n\n`)
    clients.add(res)
    // 心跳，避免中间层把空闲连接掐掉。
    const ping = setInterval(() => {
      try {
        res.write(': ping\n\n')
      } catch {
        clearInterval(ping)
      }
    }, 25_000)
    req.on('close', () => {
      clearInterval(ping)
      clients.delete(res)
    })
    return
  }

  // --- CLI 下发命令 ---
  if (url.pathname === '/command' && req.method === 'POST') {
    const body = await readBody(req)
    if (!body || typeof body.cmd !== 'string') {
      return json(res, 400, { ok: false, error: '需要 {cmd, args}' })
    }
    if (clients.size === 0) {
      return json(res, 409, {
        ok: false,
        error: '没有页面连上控制服务。请先打开工作台（npm run dev），确认右上角显示"agent 已连接"',
      })
    }
    const id = `c${++commandSeq}`
    broadcast('command', { id, cmd: body.cmd, args: body.args ?? {} })

    // 等页面回执，让 CLI 能如实告诉 agent 命令到底成没成。
    const ack = await new Promise((resolve) => {
      const timer = setTimeout(() => {
        pendingAcks.delete(id)
        resolve({ ok: false, error: '页面未在 5 秒内回执' })
      }, 5000)
      pendingAcks.set(id, { resolve, timer })
    })
    return json(res, ack.ok ? 200 : 400, ack)
  }

  // --- 浏览器回执 ---
  if (url.pathname === '/ack' && req.method === 'POST') {
    const body = await readBody(req)
    const entry = body?.id ? pendingAcks.get(body.id) : null
    if (entry) {
      clearTimeout(entry.timer)
      pendingAcks.delete(body.id)
      // 回执自带快照，顺手更新缓存，省掉一次往返。
      if (body.state) {
        latestState = body.state
        latestStateAt = Date.now()
      }
      entry.resolve({
        ok: body.ok !== false,
        error: body.error,
        result: body.result,
        state: body.state,
      })
    }
    return json(res, 200, { ok: true })
  }

  // --- 浏览器推送状态 ---
  if (url.pathname === '/state' && req.method === 'POST') {
    latestState = await readBody(req)
    latestStateAt = Date.now()
    return json(res, 200, { ok: true })
  }

  // --- CLI 读状态 ---
  if (url.pathname === '/state' && req.method === 'GET') {
    if (!latestState) {
      return json(res, 409, {
        ok: false,
        error: '还没有页面推送过状态。请确认工作台已打开并连上控制服务',
      })
    }
    return json(res, 200, {
      ok: true,
      ageMs: Date.now() - latestStateAt,
      clients: clients.size,
      state: latestState,
    })
  }

  if (url.pathname === '/health') {
    return json(res, 200, { ok: true, clients: clients.size, hasState: Boolean(latestState) })
  }

  json(res, 404, { ok: false, error: `未知路径 ${url.pathname}` })
})

server.listen(PORT, '127.0.0.1', () => {
  // 只监听回环地址：这是本地分析工具，没有理由暴露到网络上（§19 本地优先）。
  console.log(`[kxc-wb] 控制服务已启动 http://127.0.0.1:${PORT}（仅本机）`)
})
