import './tiles.css'
/**
 * Pass Waterfall：按执行顺序看每个 Pass 的耗时与 IR 体积变化。
 *
 * 查询：{kind:'pass_waterfall'}
 * 按时间顺序画横向条，用 depth 做缩进体现嵌套。
 * 条上标 IR 字节变化，changed=false 的 Pass 视觉上要弱化。
 */

import React, { useMemo } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs, formatBytes } from '../ui/chart/format'
import { useWorkbench } from '../state/store'

export interface PassWaterfallTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function PassWaterfallTile(props: PassWaterfallTileProps): JSX.Element {
  const { tile, context, density } = props
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'pass_waterfall' }, filter)

  const option = useMemo(() => {
    if (!query.data) return null

    const data = query.data
    const items = data.items

    // 构造 ECharts 数据
    const categories = items.map((_, i) => `Pass ${i}`)
    const seriesData = items.map((item) => ({
      name: item.passName,
      value: [item.startNs, item.startNs + item.durationNs],
      passName: item.passName,
      component: item.component,
      changed: item.changed,
      depth: item.depth,
      irBefore: item.irBeforeBytes,
      irAfter: item.irAfterBytes,
      itemStyle: {
        color: item.changed ? 'var(--series-1)' : 'var(--text-muted)',
        opacity: item.changed ? 1 : 0.5,
      },
    }))

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (!params.value) return ''
          const delta = params.value[1] - params.value[0]
          let html = `<strong>${params.name}</strong><br/>耗时: ${formatNs(delta)}`
          if (params.irBefore !== null && params.irAfter !== null) {
            html += `<br/>IR: ${formatBytes(params.irBefore)} → ${formatBytes(params.irAfter)}`
          }
          if (!params.changed) html += '<br/>⚠ 未改动 IR'
          return html
        },
      },
      grid: { left: 100, right: 20, top: 20, bottom: 30 },
      xAxis: {
        type: 'time',
        min: data.startNs,
        max: data.endNs,
      },
      yAxis: {
        type: 'category',
        data: categories,
        axisLabel: { formatter: (v: string) => v },
      },
      series: [
        {
          type: 'heatmap',
          data: seriesData.map((item, i) => [(item.value?.[0] ?? 0), i, ((item.value?.[1] ?? 0) - (item.value?.[0] ?? 0))]),
          visualMap: { min: 0, max: (data.endNs ?? 0) - (data.startNs ?? 0), show: false },
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      click: (params: any) => {
        const item = query.data?.items.find((it) => it.passName === params.name)
        if (item?.passName) {
          setTileSelection(tile.id, { kind: 'pass', value: item.passName })
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
    return <div className="tile-content">无 Pass 事件</div>
  }

  const data = query.data

  return (
    <div className="tile-content pass-waterfall">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '240px' }} />

      <div className="sr-only">
        Pass 瀑布，共 {data.items.length} 个 Pass，总耗时 {formatNs(data.endNs - data.startNs)}。
        {data.items.slice(0, 3).map((item) => `${item.passName} ${formatNs(item.durationNs)} ${item.changed ? '（改动 IR）' : '（未改动）'}`).join('；')}
      </div>
    </div>
  )
}
