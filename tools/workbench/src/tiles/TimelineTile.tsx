import './tiles.css'
/**
 * Timeline Tile：按时间轴查看事件分布。
 *
 * 查询：{kind:'timeline', buckets}
 * 内置轻量时间线：按 component 分道画堆叠柱。
 * 支持框选时间范围写回全局 context。
 * 另外提供"在 Perfetto 中打开"按钮，但必须先弹出确认说明数据将发送到外部站点（§19.2）。
 *
 * 重型 Tile（heavy: true）。
 */

import React, { useCallback, useMemo, useState } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter, getQueryClient } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs } from '../ui/chart/format'
import { useWorkbench } from '../state/store'

export interface TimelineTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function TimelineTile(props: TimelineTileProps): JSX.Element {
  const { tile, context, density } = props
  const [showPerfettoWarn, setShowPerfettoWarn] = useState(false)
  const setGlobalFilter = useWorkbench((s) => s.setGlobalFilter)

  const filter = toQueryFilter(context)
  const buckets = density === 'compact' ? 50 : 200
  const query = useQuery(context.bundleId, { kind: 'timeline', buckets }, filter)

  const traceText = context.bundleId ? getQueryClient().getTrace(context.bundleId) : null

  /**
   * 把 trace 交给 ui.perfetto.dev。
   *
   * Perfetto 的对接方式是固定握手：先开窗口，反复发 'PING' 直到对方回 'PONG'，
   * 再把 ArrayBuffer 通过 postMessage 送过去。不能直接用 URL 传——trace 动辄几十 MB。
   * 只有用户在上面的确认框里点了"继续"才会走到这里（§19.2：未经用户操作不得上传）。
   */
  const openInPerfetto = useCallback(() => {
    if (!traceText) return
    const win = window.open('https://ui.perfetto.dev', '_blank')
    if (!win) {
      setShowPerfettoWarn(false)
      return
    }
    const buffer = new TextEncoder().encode(traceText).buffer
    const timer = window.setInterval(() => win.postMessage('PING', 'https://ui.perfetto.dev'), 250)
    const onMessage = (ev: MessageEvent) => {
      if (ev.data !== 'PONG') return
      window.clearInterval(timer)
      window.removeEventListener('message', onMessage)
      win.postMessage(
        { perfetto: { buffer, title: `KXC ${context.bundleId ?? ''}`, fileName: 'trace.json' } },
        'https://ui.perfetto.dev',
      )
    }
    window.addEventListener('message', onMessage)
    // 对方一直不回就放弃，避免定时器泄漏。
    window.setTimeout(() => {
      window.clearInterval(timer)
      window.removeEventListener('message', onMessage)
    }, 20_000)
    setShowPerfettoWarn(false)
  }, [traceText, context.bundleId])

  const option = useMemo(() => {
    if (!query.data) return null

    const data = query.data
    const bucketCount = Math.max(...data.series.map((s) => s.values.length), data.counts.length)

    // x 轴必须是**时间桶**。旧实现把 series.values（每桶一个值）直接塞给
    // "类目=component" 的坐标系，ECharts 把第 i 个桶的值对到第 i 个 component
    // 类目上——整张图的坐标语义都是错的（Codex review 抓到）。
    // 正确形态：x=桶起始时刻，y=该桶内耗时，每个 component 一条堆叠柱序列。
    const bucketLabels = Array.from({ length: bucketCount }, (_, i) =>
      formatNs(data.startNs + i * data.bucketNs),
    )
    const compact = density === 'compact' || density === 'thumbnail'

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: compact ? undefined : { top: 0, textStyle: { fontSize: 10 } },
      grid: { left: 60, right: 16, top: compact ? 8 : 28, bottom: compact ? 20 : 44 },
      xAxis: {
        type: 'category',
        data: bucketLabels,
        axisLabel: { show: !compact, fontSize: 10, interval: Math.ceil(bucketCount / 8) },
      },
      yAxis: {
        type: 'value',
        axisLabel: { show: !compact, formatter: (v: number) => formatNs(v) },
      },
      // dataZoom 是把框选写回全局时间范围的入口（下方 datazoom 事件依赖它）。
      dataZoom: compact
        ? undefined
        : [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 4 }],
      series: data.series.map((s, idx) => ({
        name: s.component,
        type: 'bar' as const,
        stack: 'total',
        barCategoryGap: '10%',
        data: s.values,
        itemStyle: { color: `var(--series-${(idx % 8) + 1})` },
      })),
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      datazoom: (params: any) => {
        if (query.data) {
          const { start, end } = params.batch?.[0] || { start: 0, end: 100 }
          const totalNs = query.data.endNs - query.data.startNs
          const startNs = query.data.startNs + (start / 100) * totalNs
          const endNs = query.data.startNs + (end / 100) * totalNs
          setGlobalFilter({ timeRange: { startNs, endNs } })
        }
      },
    },
  })

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data) return <div className="tile-content">无时间线数据</div>

  return (
    <div className="tile-content timeline">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '260px' }} />

      {density !== 'compact' && (
        <div className="timeline-controls">
          <button
            onClick={() => setShowPerfettoWarn(true)}
            title="在 Perfetto 中查看完整 trace，但需要将数据发送到 ui.perfetto.dev"
          >
            📊 在 Perfetto 中打开
          </button>
        </div>
      )}

      {showPerfettoWarn && (
        <div className="modal-overlay" onClick={() => setShowPerfettoWarn(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
            <h3>确认发送到外部站点</h3>
            <p>
              继续后，本 bundle 的 <code>trace.json</code> 会被发送到{' '}
              <strong>ui.perfetto.dev</strong>（Google 运营的外部站点）。
              trace 里含有 pass 名、算子名与耗时，可能反映你的模型结构。
            </p>
            <p style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)' }}>
              注意：当前 trace.json 的 args 只有 status/severity/message，不含 span_id，
              因此在 Perfetto 里选中事件无法回跳到本工作台。
            </p>
            <div className="modal-actions">
              <button onClick={() => setShowPerfettoWarn(false)}>取消</button>
              <button onClick={openInPerfetto} disabled={!traceText}>
                {traceText ? '继续打开' : '该 bundle 没有 trace.json'}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="sr-only">
        时间线，事件从 {formatNs(query.data.startNs)} 到 {formatNs(query.data.endNs)}，
        共 {query.data.series.length} 个 component。
        {query.data.series.map((s) => `${s.component}: ${s.values.reduce((a, b) => a + b, 0)} ns`).join('；')}
      </div>
    </div>
  )
}
