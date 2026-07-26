import './tiles.css'
/**
 * 阶段耗时分解：编译时间花在 Relay、Lowering、TIR、CodeGen 哪一段。
 *
 * 查询：{kind:'phase_breakdown'}
 * 要显示 unattributedNs（未归类耗时），诚实呈现"阶段之和 ≠ 总耗时"。
 */

import React, { useMemo } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs } from '../ui/chart/format'

export interface PhaseBreakdownTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function PhaseBreakdownTile(props: PhaseBreakdownTileProps): JSX.Element {
  const { tile, context, density } = props

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'phase_breakdown' }, filter)

  const option = useMemo(() => {
    if (!query.data) return null

    const data = query.data
    const items = data.items.map((item) => ({
      name: `${item.component}/${item.phase}`,
      value: item.durationNs,
    }))

    // 如果有未归类耗时，也加入
    if (data.unattributedNs && data.unattributedNs > 0) {
      items.push({ name: '（未归类）', value: data.unattributedNs })
    }

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.componentSubType !== 'pie') return ''
          const percent = ((params.value / data.totalNs) * 100).toFixed(1)
          return `${params.name}<br/>${formatNs(params.value)} (${percent}%)`
        },
      },
      legend: density !== 'compact' ? { orient: 'right', type: 'scroll' } : undefined,
      series: [
        {
          type: 'pie',
          data: items,
          radius: density === 'thumbnail' ? '50%' : '70%',
          label: density !== 'thumbnail' ? { formatter: '{b}' } : false,
          emphasis: { itemStyle: { borderWidth: 2 } },
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, { theme: 'dark' })

  if (query.status === 'loading') {
    return <div className="tile-content">加载中...</div>
  }

  if (query.status === 'error') {
    return <div className="tile-content error">错误: {query.error}</div>
  }

  if (query.status === 'cancelled') {
    return <div className="tile-content">已取消</div>
  }

  if (!query.data) {
    return <div className="tile-content">无数据</div>
  }

  const data = query.data
  const hasUnattributed = data.unattributedNs && data.unattributedNs > 0
  const unattributedPct = hasUnattributed ? ((data.unattributedNs! / data.totalNs) * 100).toFixed(1) : null

  return (
    <div className="tile-content phase-breakdown">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '200px' }} />

      {hasUnattributed && (
        <div className="warning" role="alert">
          ⚠ 存在 {formatNs(data.unattributedNs)} ({unattributedPct}%) 未归类耗时
        </div>
      )}

      <div className="sr-only">
        阶段耗时分解，共 {formatNs(data.totalNs)}。
        {data.items.map((item) => `${item.component}/${item.phase}: ${formatNs(item.durationNs)}`).join('；')}
        {hasUnattributed && `；未归类: ${formatNs(data.unattributedNs)}`}
      </div>
    </div>
  )
}
