import './tiles.css'
/**
 * Shape × Cache 热力图：哪些输入 shape 反复未命中缓存、触发同步编译。
 *
 * 查询：{kind:'shape_cache'}
 * 行=shape 签名，列=exact/fuzzy/miss。
 * unresolved > 0 时提示有多少缓存事件无法解析出 shape。
 */

import React, { useMemo } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatCount } from '../ui/chart/format'
import { useWorkbench } from '../state/store'

export interface ShapeCacheHeatmapTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function ShapeCacheHeatmapTile(props: ShapeCacheHeatmapTileProps): JSX.Element {
  const { tile, context, density } = props
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'shape_cache' }, filter)

  const option = useMemo(() => {
    if (!query.data) return null

    const cells = query.data.cells.slice(0, density === 'compact' ? 10 : 30)
    const shapes = cells.map((c) => c.shapeSignature)
    const categories = ['exact', 'fuzzy', 'miss']

    const data: any[] = []
    cells.forEach((cell, shapeIdx) => {
      data.push([0, shapeIdx, cell.exactHit])
      data.push([1, shapeIdx, cell.fuzzyHit])
      data.push([2, shapeIdx, cell.miss])
    })

    const maxVal = Math.max(...cells.map((c) => Math.max(c.exactHit, c.fuzzyHit, c.miss)))

    return {
      tooltip: { trigger: 'item', formatter: (params: any) => `${params.name}: ${params.value[2]}` },
      grid: { left: 150, right: 20, top: 20, bottom: 20 },
      xAxis: { type: 'category', data: categories },
      yAxis: { type: 'category', data: shapes, axisLabel: { fontSize: density === 'compact' ? 10 : 11 } },
      visualMap: { min: 0, max: maxVal, inRange: { color: ['#1a2028', 'var(--series-1)'] } },
      series: [
        {
          type: 'heatmap',
          data: data,
          itemStyle: { borderWidth: 1, borderColor: 'var(--border)' },
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      click: (params: any) => {
        const cell = query.data?.cells.find((c) => c.shapeSignature === params.name)
        if (cell) setTileSelection(tile.id, { kind: 'shape', value: cell.shapeSignature })
      },
    },
  })

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data || query.data.cells.length === 0) return <div className="tile-content">无缓存数据</div>

  const data = query.data

  return (
    <div className="tile-content shape-cache-heatmap">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '220px' }} />

      {data.unresolved > 0 && (
        <div className="warning" role="alert">
          ⚠ {formatCount(data.unresolved)} 个缓存事件无法解析出 shape（父 span 缺失）
        </div>
      )}

      <div className="sr-only">
        Shape × Cache 热力图，共 {formatCount(data.cells.length)} 个 shape。
        {data.unresolved > 0 && `有 ${formatCount(data.unresolved)} 个缓存事件无法解析。`}
        最多未命中的 shape 是 {data.cells[0]?.shapeSignature}
      </div>
    </div>
  )
}
