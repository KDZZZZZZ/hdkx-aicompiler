import './tiles.css'
/**
 * 回归对比：Baseline 与 Candidate 的逐 Pass 耗时差异。
 *
 * 查询：两个 bundle 的 {kind:'pass_ranking'}
 * 对不齐的对象标记（例如 baseline 有而 candidate 没有的 Pass）。
 */

import React, { useMemo } from 'react'
import * as echarts from 'echarts'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useECharts } from '../ui/chart/useECharts'
import { formatNs, formatDelta } from '../ui/chart/format'
import { useWorkbench } from '../state/store'

export interface RegressionDeltaTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function RegressionDeltaTile(props: RegressionDeltaTileProps): JSX.Element {
  const { tile, context, density } = props
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter(context)
  const baselineFilter = { ...filter, ... context.baselineBundleId ? { bundleId: context.baselineBundleId } : {} }
  const candidateFilter = { ...filter, ... context.bundleId ? { bundleId: context.bundleId } : {} }

  // limit 要足够大：截断必须发生在**两侧对齐之后**。
  // 之前两边各取 top20 再按 Set 插入序 slice(0,15)，baseline 的 key 恒排在前，
  // candidate 侧新出现的大回归——恰恰是最该看的——会被挤出显示范围（Codex review 抓到）。
  const baselineQuery = useQuery(context.baselineBundleId, { kind: 'pass_ranking', limit: 500 }, baselineFilter)
  const candidateQuery = useQuery(context.bundleId, { kind: 'pass_ranking', limit: 500 }, candidateFilter)

  const option = useMemo(() => {
    if (!baselineQuery.data || !candidateQuery.data) return null

    const baseline = new Map(baselineQuery.data.items.map((it) => [it.passName, it]))
    const candidate = new Map(candidateQuery.data.items.map((it) => [it.passName, it]))
    const allPasses = new Set([...baseline.keys(), ...candidate.keys()])
    // 全量对齐后按差异绝对值降序，取影响最大的展示
    const shown = density === 'compact' || density === 'thumbnail' ? 8 : 15
    const passes = Array.from(allPasses)
      .sort((x, y) => {
        const dx = Math.abs((candidate.get(x)?.totalNs ?? 0) - (baseline.get(x)?.totalNs ?? 0))
        const dy = Math.abs((candidate.get(y)?.totalNs ?? 0) - (baseline.get(y)?.totalNs ?? 0))
        return dy - dx
      })
      .slice(0, shown)

    const baselineData: number[] = []
    const candidateData: number[] = []
    const status: string[] = []

    passes.forEach((pass) => {
      const b = baseline.get(pass)
      const c = candidate.get(pass)
      baselineData.push(b?.totalNs ?? 0)
      candidateData.push(c?.totalNs ?? 0)
      if (!b) status.push('missing')
      else if (!c) status.push('missing')
      else if (c.totalNs > b.totalNs * 1.1) status.push('regress')
      else if (c.totalNs < b.totalNs * 0.9) status.push('improve')
      else status.push('ok')
    })

    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 80, right: 20, top: 20, bottom: 30 },
      xAxis: { type: 'category', data: passes },
      yAxis: { type: 'value', axisLabel: { formatter: (v: number) => formatNs(v) } },
      series: [
        {
          name: 'Baseline',
          type: 'bar',
          data: baselineData,
          itemStyle: { color: 'var(--cmp-baseline)' },
        },
        {
          name: 'Candidate',
          type: 'bar',
          data: candidateData,
          itemStyle: { color: 'var(--cmp-candidate)' },
        },
      ],
    } as echarts.EChartsOption
  }, [baselineQuery.data, candidateQuery.data, density])

  const containerRef = useECharts(option, {
    onEvents: {
      click: (params: any) => {
        setTileSelection(tile.id, { kind: 'pass', value: params.name })
      },
    },
  })

  if (baselineQuery.status === 'loading' || candidateQuery.status === 'loading') {
    return <div className="tile-content">加载中...</div>
  }

  if (baselineQuery.status === 'error' || candidateQuery.status === 'error') {
    return <div className="tile-content error">错误加载 bundle</div>
  }

  if (!baselineQuery.data || !candidateQuery.data) {
    return <div className="tile-content">无对比数据</div>
  }

  return (
    <div className="tile-content regression-delta">
      <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '220px' }} />

      <div className="sr-only">
        回归对比：Baseline vs Candidate。
        Baseline 共 {baselineQuery.data.items.length} Pass，
        Candidate 共 {candidateQuery.data.items.length} Pass。
      </div>
    </div>
  )
}
