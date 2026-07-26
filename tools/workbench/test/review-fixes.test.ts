// @vitest-environment node
/**
 * PR review（Codex）指出的缺陷的回归测试。
 * 每个用例对应一条已确认属实的评论，防止同类问题回潮。
 */

import { describe, it, expect } from 'vitest'
import { lcsDiff } from '../src/ui/chart/line-diff'
import { makeDirectoryBundleId } from '../src/kxc/bundle-source'
import { parseBundle, runQuery } from '../src/kxc/query.worker'
import type { KernelDurationDistResult } from '../src/kxc/query-protocol'

describe('lcsDiff：顶部一处改动不得把整个文件标成增删', () => {
  it('中段单行修改，前后未变行保持 unchanged', () => {
    const before = ['a', 'b', 'OLD', 'c', 'd', 'e']
    const after = ['a', 'b', 'NEW', 'c', 'd', 'e']
    const d = lcsDiff(before, after)
    // 旧实现会把 OLD c d e 全标删除、NEW c d e 全标新增
    expect(d.filter((x) => x.status === 0).map((x) => x.line)).toEqual(['a', 'b', 'c', 'd', 'e'])
    expect(d.filter((x) => x.status === -1).map((x) => x.line)).toEqual(['OLD'])
    expect(d.filter((x) => x.status === 1).map((x) => x.line)).toEqual(['NEW'])
  })

  it('穿插的多处修改也能对齐公共行', () => {
    const before = ['k1', 'x', 'k2', 'y', 'k3']
    const after = ['k1', 'X2', 'k2', 'Y2', 'k3', 'tail']
    const d = lcsDiff(before, after)
    expect(d.filter((x) => x.status === 0).map((x) => x.line)).toEqual(['k1', 'k2', 'k3'])
    expect(d.filter((x) => x.status === 1).map((x) => x.line)).toEqual(['X2', 'Y2', 'tail'])
  })

  it('空输入与全等输入', () => {
    expect(lcsDiff([], [])).toEqual([])
    expect(lcsDiff(['a'], ['a'])).toEqual([{ line: 'a', status: 0 }])
  })
})

describe('makeDirectoryBundleId：同名目录不得撞 ID', () => {
  const manifestA = JSON.stringify({ trace_id: 'trace-aaa' })
  const manifestB = JSON.stringify({ trace_id: 'trace-bbb' })

  it('有 manifest 时用 trace_id，两个同名目录 ID 不同', () => {
    const a = makeDirectoryBundleId('profile', manifestA, 'events-a')
    const b = makeDirectoryBundleId('profile', manifestB, 'events-b')
    expect(a).not.toBe(b)
    expect(a).toContain('trace-aaa')
  })

  it('无 manifest 时退化为内容指纹，内容不同则 ID 不同', () => {
    const a = makeDirectoryBundleId('profile', null, 'events-content-1')
    const b = makeDirectoryBundleId('profile', null, 'events-content-2')
    expect(a).not.toBe(b)
  })

  it('同一目录重复加载 ID 稳定（幂等缓存仍然生效）', () => {
    expect(makeDirectoryBundleId('p', manifestA, 'x')).toBe(makeDirectoryBundleId('p', manifestA, 'x'))
    expect(makeDirectoryBundleId('p', null, 'x')).toBe(makeDirectoryBundleId('p', null, 'x'))
  })
})

describe('kernel 分布：CUPTI 与逻辑 span 不得混进同一直方图', () => {
  function makeEvents(lines: Array<{ type: string; dur: number }>): string {
    return lines
      .map((l, i) =>
        JSON.stringify({
          trace_id: 't',
          session_id: 's',
          run_id: 'r',
          span_id: `sp${i}`,
          parent_span_id: '',
          component: l.type === 'cuda_kernel' ? 'backend.cuda' : 'execution_plan',
          event_type: l.type,
          phase: 'complete',
          ts_ns: i * 1000,
          duration_ns: l.dur,
          status: 'ok',
          severity: 'info',
          device: '',
          worker_id: -1,
          op_name: '',
          pass_name: '',
          kernel_symbol: '',
          shape_signature: '',
          message: '',
          fields: {},
          metrics: {},
        }),
      )
      .join('\n')
  }

  const payload = (events: string) => ({
    manifest: null,
    events,
    summary: null,
    diagnosis: null,
    trace: null,
    artifacts: {},
  })

  it('两类事件并存时只统计 CUPTI，caveat 为空', () => {
    // CUPTI 三条 100ns；逻辑 span 三条 1e6ns——混进来会把 p50 拉到六个数量级外
    const b = parseBundle('mix', payload(makeEvents([
      { type: 'cuda_kernel', dur: 100 },
      { type: 'cuda_kernel', dur: 110 },
      { type: 'cuda_kernel', dur: 120 },
      { type: 'kernel_exec', dur: 1_000_000 },
      { type: 'kernel_exec', dur: 1_100_000 },
      { type: 'kernel_exec', dur: 1_200_000 },
    ])))
    const r = runQuery(b, { kind: 'kernel_duration_dist', buckets: 4 }, {}) as KernelDurationDistResult
    expect(r.totalCount).toBe(3)
    expect(r.maxNs).toBe(120)
    expect(r.caveat).toBeNull()
  })

  it('只有逻辑 span 时使用之并保留 caveat', () => {
    const b = parseBundle('logical', payload(makeEvents([
      { type: 'kernel_exec', dur: 500 },
      { type: 'kernel_exec', dur: 700 },
    ])))
    const r = runQuery(b, { kind: 'kernel_duration_dist', buckets: 4 }, {}) as KernelDurationDistResult
    expect(r.totalCount).toBe(2)
    expect(r.caveat).toContain('CUPTI')
  })
})
