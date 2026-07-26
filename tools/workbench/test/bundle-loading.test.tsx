/**
 * Bundle 加载入口测试。
 *
 * 这一层此前是断的：useBundleActions 写好了却没有任何组件使用，
 * 界面上根本没有加载 bundle 的按钮——功能"实现了"但用户点不到。
 * 这里同时锁住两件事：IO 层能把 bundle 读成 LoadPayload，
 * 以及顶栏确实把入口暴露出来了。
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { createRoot, type Root } from 'react-dom/client'
import { act } from 'react'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { loadFromUrl, listFixtures } from '../src/kxc/bundle-source'

const FIXTURE_DIR = join(__dirname, '..', 'fixtures', 'bundles', 'baseline')

vi.mock('echarts', () => {
  const chart = {
    setOption: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    getZr: () => ({ on: vi.fn(), off: vi.fn() }),
  }
  return { init: vi.fn(() => chart), dispose: vi.fn(), use: vi.fn(), default: { init: vi.fn(() => chart) } }
})
// Worker 替身在 test/setup.ts 里直接赋值，不用 stubGlobal——
// 这里的 unstubAllGlobals() 会把 stubGlobal 注册的东西一并清掉。

/** 用真实 fixture 文件充当 HTTP 响应，避免测的是一份编出来的假数据。 */
function stubFetchFromDisk() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const name = String(url).split('/').pop() ?? ''
      if (name === 'index.json') {
        const text = readFileSync(join(FIXTURE_DIR, '..', 'index.json'), 'utf8')
        return { ok: true, text: async () => text, json: async () => JSON.parse(text) }
      }
      try {
        const text = readFileSync(join(FIXTURE_DIR, name), 'utf8')
        return { ok: true, text: async () => text, json: async () => JSON.parse(text) }
      } catch {
        return { ok: false, text: async () => '', json: async () => ({}) }
      }
    }),
  )
}

describe('bundle IO', () => {
  beforeEach(stubFetchFromDisk)
  afterEach(() => vi.unstubAllGlobals())

  it('listFixtures 能读出样例索引，且包含真实产物 real-compile', async () => {
    const list = await listFixtures()
    expect(list.length).toBeGreaterThan(0)
    expect(list.some((f) => f.id === 'real-compile' && f.variant === 'real')).toBe(true)
  })

  it('loadFromUrl 组装出的 payload 含 events 与其余 bundle 文件', async () => {
    const res = await loadFromUrl('/bundles/baseline', 'baseline')
    expect(res.payload.events.length).toBeGreaterThan(0)
    expect(res.payload.manifest).not.toBeNull()
    expect(res.payload.summary).not.toBeNull()
    expect(res.payload.diagnosis).not.toBeNull()
    expect(res.payload.trace).not.toBeNull()
    // §19.3：来源必须能告诉用户数据在哪，顶栏要显示它
    expect(res.ref.hint).toContain('本地')
  })

  it('events.jsonl 缺失时给出可读错误，而不是静默产出空 bundle', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, text: async () => '' })))
    await expect(loadFromUrl('/bundles/nope', 'nope')).rejects.toThrow(/events\.jsonl/)
  })
})

describe('顶栏暴露了加载入口', () => {
  let container: HTMLDivElement
  let root: Root

  beforeEach(() => {
    stubFetchFromDisk()
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
  })
  afterEach(() => {
    act(() => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
  })

  it('渲染出 Bundle 选择器、打开目录按钮、Baseline 选择器与可用搜索框', async () => {
    const { default: App } = await import('../src/App')
    await act(async () => {
      root.render(<App />)
    })

    expect(container.querySelector('[aria-label="选择当前 Bundle"]')).not.toBeNull()
    expect(container.querySelector('[aria-label="打开本地 bundle 目录"]')).not.toBeNull()
    expect(container.querySelector('[aria-label="选择 Baseline Bundle"]')).not.toBeNull()

    // 搜索框必须是可用的：之前它是 disabled 且写着"未实现"
    const search = container.querySelector<HTMLInputElement>('[aria-label="全局搜索"]')
    expect(search).not.toBeNull()
    expect(search!.disabled).toBe(false)
    expect(search!.placeholder).not.toContain('未实现')
  })

  it('样例列表加载后出现在 Bundle 下拉里', async () => {
    const { default: App } = await import('../src/App')
    await act(async () => {
      root.render(<App />)
    })
    // 等 listFixtures 的微任务落地
    await act(async () => {
      await Promise.resolve()
    })
    const select = container.querySelector<HTMLSelectElement>('[aria-label="选择当前 Bundle"]')!
    expect(select.textContent).toContain('real-compile')
  })
})
