import './tiles.css'
/**
 * Pass 耗时排行：找出最慢的 Pass 以及它是否真的改动了 IR。
 *
 * 查询：{kind:'pass_ranking', limit}
 */

import React, { useMemo } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs, formatBytes, formatCount } from '../ui/chart/format'
import { useWorkbench } from '../state/store'

export interface PassRankingTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function PassRankingTile(props: PassRankingTileProps): JSX.Element {
  const { tile, context, density } = props
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'pass_ranking', limit: 10 }, filter)

  const option = useMemo(() => {
    if (!query.data) return null

    const items = query.data.items.slice(0, density === 'compact' ? 5 : 10)

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 80, right: 20, top: 20, bottom: 30 },
      xAxis: { type: 'category', data: items.map((it) => it.passName) },
      yAxis: { type: 'value', axisLabel: { formatter: (v: number) => formatNs(v) } },
      series: [
        {
          type: 'bar',
          data: items.map((it) => ({
            value: it.totalNs,
            itemStyle: { color: it.changed ? 'var(--series-1)' : 'var(--text-muted)', opacity: it.changed ? 1 : 0.5 },
          })),
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      click: (params: any) => {
        const item = query.data?.items.find((it) => it.passName === params.name)
        if (item) setTileSelection(tile.id, { kind: 'pass', value: item.passName })
      },
    },
  })

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data || query.data.items.length === 0) return <div className="tile-content">无数据</div>

  const data = query.data

  return (
    <div className="tile-content pass-ranking">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '200px' }} />

      <div className="sr-only">
        Pass 耗时排行，共 {formatCount(data.items.length)} 个 Pass，总耗时 {formatNs(data.totalNs)}。
        最慢的 Pass 是 {data.items[0]?.passName} {data.items[0] && formatNs(data.items[0].totalNs)}
        {data.items[0]?.changed ? '（改动 IR）' : '（未改动 IR）'}
      </div>
    </div>
  )
}
