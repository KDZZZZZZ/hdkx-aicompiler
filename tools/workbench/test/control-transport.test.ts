// @vitest-environment node
/**
 * 控制链路端到端测试：CLI → 控制服务 → 页面 → 回执。
 *
 * 用一个模拟页面（裸 HTTP 订阅 SSE）代替真浏览器，这样传输契约本身可以在
 * 没有浏览器的环境里被锁住。浏览器侧的命令映射由 app-smoke 覆盖，
 * 两边合起来才算这条链路被验证过。
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { spawn, type ChildProcess } from 'node:child_process'
import { get, request } from 'node:http'
import { join } from 'node:path'

const PORT = 5399
const BASE = `http://127.0.0.1:${PORT}`
const SERVER = join(__dirname, '..', 'server', 'control-server.mjs')

let server: ChildProcess

function post(path: string, body: unknown): Promise<{ status: number; body: any }> {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body)
    const req = request(
      `${BASE}${path}`,
      { method: 'POST', headers: { 'content-type': 'application/json' } },
      (res) => {
        let data = ''
        res.on('data', (c) => (data += c))
        res.on('end', () => resolve({ status: res.statusCode ?? 0, body: JSON.parse(data || '{}') }))
      },
    )
    req.on('error', reject)
    req.end(payload)
  })
}

function getJson(path: string): Promise<{ status: number; body: any }> {
  return new Promise((resolve, reject) => {
    get(`${BASE}${path}`, (res) => {
      let data = ''
      res.on('data', (c) => (data += c))
      res.on('end', () => resolve({ status: res.statusCode ?? 0, body: JSON.parse(data || '{}') }))
    }).on('error', reject)
  })
}

/** 模拟页面：订阅 SSE，收到命令就按 handler 回执。 */
function fakePage(
  handler: (
    cmd: string,
    args: any,
  ) => { ok: boolean; error?: string; result?: unknown; state?: unknown } | null,
) {
  return new Promise<{ close: () => void }>((resolve, reject) => {
    const req = get(`${BASE}/events`, (res) => {
      let buf = ''
      res.on('data', (chunk) => {
        buf += chunk
        // SSE 帧以空行分隔
        let idx: number
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          const evLine = frame.split('\n').find((l) => l.startsWith('event: '))
          const dataLine = frame.split('\n').find((l) => l.startsWith('data: '))
          if (!evLine || !dataLine) continue
          const event = evLine.slice(7)
          const data = JSON.parse(dataLine.slice(6))
          if (event === 'hello') resolve({ close: () => req.destroy() })
          if (event === 'command') {
            const r = handler(data.cmd, data.args)
            // handler 返回 null 表示故意不回执，用来验证服务端的超时保护。
            if (r) void post('/ack', { id: data.id, ...r })
          }
        }
      })
    })
    req.on('error', reject)
  })
}

/** 等到服务端记录的连接数正好是 n。close() 后连接是异步摘除的，
 *  不加这个同步点就会出现"命令被广播给上一个还没断干净的页面"的竞态。 */
async function waitForClients(n: number): Promise<void> {
  await waitFor(async () => (await getJson('/health')).body.clients === n)
}

async function waitFor(fn: () => Promise<boolean>, ms = 4000): Promise<void> {
  const start = Date.now()
  while (Date.now() - start < ms) {
    if (await fn()) return
    await new Promise((r) => setTimeout(r, 60))
  }
  throw new Error('等待超时')
}

beforeAll(async () => {
  server = spawn(process.execPath, [SERVER], {
    env: { ...process.env, KXC_WB_PORT: String(PORT) },
    stdio: 'ignore',
  })
  await waitFor(async () => {
    try {
      return (await getJson('/health')).status === 200
    } catch {
      return false
    }
  })
})

afterAll(() => {
  server?.kill()
})

describe('控制服务', () => {
  it('没有页面连接时拒绝命令，并说清该怎么办', async () => {
    const res = await post('/command', { cmd: 'add-tile', args: { type: 'kpi' } })
    expect(res.status).toBe(409)
    expect(res.body.error).toContain('没有页面连上')
  })

  it('命令能送达页面，回执原样返回给调用方', async () => {
    const seen: Array<{ cmd: string; args: any }> = []
    const page = await fakePage((cmd, args) => {
      seen.push({ cmd, args })
      return { ok: true, result: { tileId: 'tile-x' }, state: { mode: 'strip', columns: [] } }
    })

    const res = await post('/command', { cmd: 'add-tile', args: { type: 'kpi' } })
    expect(res.status).toBe(200)
    expect(res.body.ok).toBe(true)
    expect(res.body.result).toEqual({ tileId: 'tile-x' })
    expect(seen).toEqual([{ cmd: 'add-tile', args: { type: 'kpi' } }])

    // 回执自带的快照必须立刻可读——agent 执行完命令不应再读到旧状态
    const st = await getJson('/state')
    expect(st.body.state).toEqual({ mode: 'strip', columns: [] })
    page.close()
    await waitForClients(0)
  })

  it('页面报错时如实回传错误，不伪装成成功', async () => {
    const page = await fakePage(() => ({ ok: false, error: '未知图表类型 nope' }))
    const res = await post('/command', { cmd: 'add-tile', args: { type: 'nope' } })
    expect(res.status).toBe(400)
    expect(res.body.ok).toBe(false)
    expect(res.body.error).toContain('未知图表类型')
    page.close()
    await waitForClients(0)
  })

  it('页面不回执时超时返回，不会把调用方永久挂起', async () => {
    // 命令是广播给所有在线页面的，所以必须先确认没有别的页面残留，
    // 否则上一个用例的页面会替这条命令回执，测不到超时分支。
    await waitForClients(0)
    const silent = await fakePage(() => null)
    await waitForClients(1)

    const res = await post('/command', { cmd: 'undo', args: {} })
    expect(res.body.ok).toBe(false)
    expect(res.body.error).toContain('回执')
    silent.close()
  }, 15_000)

  it('只监听回环地址，不暴露到网络', async () => {
    const health = await getJson('/health')
    expect(health.body.ok).toBe(true)
    // server.listen 绑定 127.0.0.1；这里断言配置意图，防止以后被改成 0.0.0.0
    const src = await import('node:fs').then((fs) =>
      fs.readFileSync(SERVER, 'utf8'),
    )
    expect(src).toContain("'127.0.0.1'")
    expect(src).not.toContain("'0.0.0.0'")
  })
})
