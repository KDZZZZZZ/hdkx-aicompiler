import './tiles.css'
/**
 * Kernel Top-N：最耗时的 kernel 排行。
 *
 * 查询：隐含 {kind:'hotspot', by:'kernel', limit:10}
 * readiness 为 partial：CPU 路径下 kernel_symbol 多为空。
 */

import React, { useMemo } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs, formatCount } from '../ui/chart/format'
import { useWorkbench } from '../state/store'

export interface KernelTopNTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function KernelTopNTile(props: KernelTopNTileProps): JSX.Element {
  const { tile, context, density } = props
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'hotspot', by: 'kernel', limit: 10 }, filter)

  const option = useMemo(() => {
    if (!query.data) return null

    const items = query.data.items.slice(0, density === 'compact' ? 5 : 10)

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 100, right: 20, top: 20, bottom: 30 },
      xAxis: { type: 'category', data: items.map((it) => it.key.slice(0, 30)) },
      yAxis: { type: 'value', axisLabel: { formatter: (v: number) => formatNs(v) } },
      series: [
        {
          type: 'bar',
          data: items.map((it) => ({ value: it.totalNs, name: it.key })),
          itemStyle: { color: 'var(--series-2)' },
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      click: (params: any) => {
        const item = query.data?.items.find((it) => it.key === params.name)
        if (item) setTileSelection(tile.id, { kind: 'kernel', value: item.key })
      },
    },
  })

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data || query.data.items.length === 0) return <div className="tile-content">无 Kernel 数据</div>

  const data = query.data

  return (
    <div className="tile-content kernel-topn">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '200px' }} />

      {data.caveat && (
        <div className="warning" role="alert">
          ⚠ {data.caveat}
        </div>
      )}

      <div className="sr-only">
        Kernel Top-N，共 {formatCount(data.items.length)} 个 kernel。
        最慢的是 {data.items[0]?.key} {data.items[0] && formatNs(data.items[0].totalNs)}
      </div>
    </div>
  )
}
