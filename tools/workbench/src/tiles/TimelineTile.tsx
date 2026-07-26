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

import React, { useMemo, useState } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
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

  const option = useMemo(() => {
    if (!query.data) return null

    const data = query.data
    const numBuckets = Math.ceil((data.endNs - data.startNs) / data.bucketNs)

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'line' } },
      grid: { left: 80, right: 20, top: 20, bottom: 30 },
      xAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => formatNs(v) },
      },
      yAxis: { type: 'category', data: data.series.map((s) => s.component) },
      series: data.series.map((s) => ({
        name: s.component,
        type: 'bar',
        stack: true,
        data: s.values,
        itemStyle: { color: `var(--series-${(data.series.indexOf(s) % 8) + 1})` },
      })),
    } as echarts.EChartsOption
  }, [query.data])

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
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>确认发送到外部站点</h3>
            <p>
              点击后，您的 trace.json 将被发送到 <strong>ui.perfetto.dev</strong>（外部站点）。
              请确保您已充分了解数据外流的风险，并同意此操作。
            </p>
            <div className="modal-actions">
              <button onClick={() => setShowPerfettoWarn(false)}>取消</button>
              <button
                onClick={() => {
                  // 实际实现时在这里打开 Perfetto，但当前没有 trace.json 的实际链接
                  alert('打开 Perfetto... (功能待完成)')
                  setShowPerfettoWarn(false)
                }}
              >
                继续打开
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
