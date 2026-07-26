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
    if (items.length === 0) return null

    // 瀑布图 = 两段水平柱：一段透明占位把柱子推到起始时刻，一段是真实耗时。
    // （早先这里错用了 heatmap 系列，还把 visualMap 塞进 series 里，
    //  ECharts 直接抛异常，把整个工作台带白屏了。）
    const css = getComputedStyle(document.documentElement)
    const colorChanged = css.getPropertyValue('--series-1').trim() || '#5b9dd9'
    const colorUnchanged = css.getPropertyValue('--text-muted').trim() || '#67727f'
    const axisColor = css.getPropertyValue('--text-secondary').trim() || '#98a4b3'

    const labels = items.map((it) => {
      // 用缩进体现 span 嵌套层级
      const indent = '  '.repeat(Math.min(it.depth, 4))
      return `${indent}${it.passName}`
    })

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const idx = Array.isArray(params) ? params[0]?.dataIndex : params.dataIndex
          const it = items[idx]
          if (!it) return ''
          let html = `<strong>${it.passName}</strong><br/>耗时 ${formatNs(it.durationNs)}`
          if (it.irBeforeBytes != null && it.irAfterBytes != null) {
            html += `<br/>IR ${formatBytes(it.irBeforeBytes)} → ${formatBytes(it.irAfterBytes)}`
          }
          // 花了时间却没改 IR，本身就是个值得注意的结论
          if (!it.changed) html += '<br/>⚠ 未改动 IR'
          if (it.status !== 'ok') html += `<br/>状态 ${it.status}`
          return html
        },
      },
      grid: {
        left: density === 'compact' || density === 'thumbnail' ? 8 : 140,
        right: 16,
        top: 8,
        bottom: 24,
        containLabel: density === 'compact' || density === 'thumbnail',
      },
      xAxis: {
        type: 'value',
        min: 0,
        max: Math.max(1, data.endNs - data.startNs),
        axisLabel: {
          color: axisColor,
          formatter: (v: number) => formatNs(v),
          show: density !== 'thumbnail',
        },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'category',
        data: labels,
        inverse: true,
        axisLabel: {
          color: axisColor,
          show: density !== 'compact' && density !== 'thumbnail',
          fontSize: 11,
        },
        axisTick: { show: false },
      },
      series: [
        {
          // 透明占位段
          type: 'bar',
          stack: 'wf',
          silent: true,
          itemStyle: { color: 'transparent' },
          data: items.map((it) => it.startNs - data.startNs),
        },
        {
          type: 'bar',
          stack: 'wf',
          data: items.map((it) => ({
            value: Math.max(it.durationNs, 1),
            // 未改动 IR 的 Pass 视觉上弱化
            itemStyle: {
              color: it.changed ? colorChanged : colorUnchanged,
              opacity: it.changed ? 1 : 0.45,
            },
          })),
          barMaxWidth: 14,
        },
      ],
    } as echarts.EChartsOption
  }, [query.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      click: (params: any) => {
        // y 轴标签带缩进，不能拿 params.name 去匹配，用下标定位。
        const item = query.data?.items[params.dataIndex]
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
