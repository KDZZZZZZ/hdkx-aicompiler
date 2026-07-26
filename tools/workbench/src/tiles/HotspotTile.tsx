import './tiles.css'
/**
 * Hotspot Top-N：按 Pass / 算子 / Kernel / Component 排出最耗时的对象。
 *
 * 查询：{kind:'hotspot', by, limit}，by 可在 pass/op/kernel/component 间切换。
 * by='op' 时结果带 caveat（数据来自 fields.op_name 非真实执行），必须显示。
 */

import React, { useMemo, useState } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs, formatCount } from '../ui/chart/format'
import { useWorkbench } from '../state/store'

export interface HotspotTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

type HotspotBy = 'pass' | 'op' | 'kernel' | 'component'

export function HotspotTile(props: HotspotTileProps): JSX.Element {
  const { tile, context, density } = props
  const [by, setBy] = useState<HotspotBy>('pass')
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'hotspot', by, limit: density === 'thumbnail' ? 5 : 10 }, filter)

  const option = useMemo(() => {
    if (!query.data) return null

    const data = query.data
    const items = data.items.slice(0, density === 'compact' ? 5 : 10)

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: density === 'compact' ? 60 : 80, right: 20, bottom: 30, top: 20 },
      xAxis: { type: 'category' },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => formatNs(v) },
      },
      series: [
        {
          type: 'bar',
          data: items.map((item) => ({
            value: item.totalNs,
            name: item.key,
          })),
          itemStyle: { color: 'var(--series-1)' },
          label: density !== 'thumbnail' ? { show: false } : undefined,
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      click: (params: any) => {
        const itemData = query.data?.items.find((item) => item.key === params.name)
        if (itemData) {
          setTileSelection(tile.id, { kind: by as any, value: itemData.key })
        }
      },
    },
  })

  if (query.status === 'loading') {
    return <div className="tile-content">加载中...</div>
  }

  if (query.status === 'error') {
    return <div className="tile-content error">错误: {query.error}</div>
  }

  if (query.status === 'cancelled') {
    return <div className="tile-content">已取消</div>
  }

  if (!query.data || query.data.items.length === 0) {
    return <div className="tile-content">无数据</div>
  }

  const data = query.data

  return (
    <div className="tile-content hotspot">
      {density !== 'thumbnail' && (
        <div className="controls">
          <select value={by} onChange={(e) => setBy(e.target.value as HotspotBy)}>
            <option value="pass">按 Pass</option>
            <option value="op">按算子</option>
            <option value="kernel">按 Kernel</option>
            <option value="component">按 Component</option>
          </select>
        </div>
      )}

      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '200px' }} />

      {data.caveat && (
        <div className="warning" role="alert">
          ⚠ {data.caveat}
        </div>
      )}

      <div className="sr-only">
        Hotspot by {by}，共 {formatCount(data.items.length)} 项。
        前 5：{data.items.slice(0, 5).map((item) => `${item.key} ${formatNs(item.totalNs)}`).join('；')}
      </div>
    </div>
  )
}
