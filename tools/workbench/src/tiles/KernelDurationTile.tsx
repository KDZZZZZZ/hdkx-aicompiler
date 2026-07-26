import './tiles.css'
/**
 * Kernel 耗时分布：看 kernel 耗时的分布形态与长尾。
 *
 * 查询：{kind:'kernel_duration_dist', buckets}
 * 显示 p50/p95/p99，无 CUPTI 时只有逻辑执行耗时。
 */

import React, { useMemo } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs, formatCount } from '../ui/chart/format'

export interface KernelDurationTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function KernelDurationTile(props: KernelDurationTileProps): JSX.Element {
  const { tile, context, density } = props

  const filter = toQueryFilter(context)
  const buckets = density === 'compact' ? 8 : 16
  const query = useQuery(context.bundleId, { kind: 'kernel_duration_dist', buckets }, filter)

  const option = useMemo(() => {
    if (!query.data) return null

    const data = query.data
    const bucketLabels = data.buckets.map((b) => `${formatNs(b.lowNs)}-${formatNs(b.highNs)}`)

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 80, right: 20, top: 20, bottom: 30 },
      xAxis: { type: 'category', data: bucketLabels, axisLabel: { rotate: 45 } },
      yAxis: { type: 'value', axisLabel: { formatter: (v: number) => formatCount(v) } },
      series: [
        {
          type: 'bar',
          data: data.buckets.map((b) => b.count),
          itemStyle: { color: 'var(--series-1)' },
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option)

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data || query.data.totalCount === 0) return <div className="tile-content">无 Kernel 数据</div>

  const data = query.data

  return (
    <div className="tile-content kernel-duration-dist">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '200px' }} />

      {density !== 'compact' && (
        <div className="stats">
          {data.p50Ns && <span>p50: {formatNs(data.p50Ns)}</span>}
          {data.p95Ns && <span>p95: {formatNs(data.p95Ns)}</span>}
          {data.maxNs && <span>max: {formatNs(data.maxNs)}</span>}
        </div>
      )}

      {data.caveat && (
        <div className="warning" role="alert">
          ⚠ {data.caveat}
        </div>
      )}

      <div className="sr-only">
        Kernel 耗时分布，共 {formatCount(data.totalCount)} 个 kernel。
        p50 {data.p50Ns && formatNs(data.p50Ns)}，
        p95 {data.p95Ns && formatNs(data.p95Ns)}，
        max {data.maxNs && formatNs(data.maxNs)}
      </div>
    </div>
  )
}
