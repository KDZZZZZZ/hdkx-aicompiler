import './tiles.css'
/**
 * KPI Tile：一屏看清编译总耗时、Pass 数、缓存命中率与告警数。
 *
 * 查询：{kind:'kpi'}
 * 特殊处理：cacheHitRate 分母为 0 时显示 "—" 而非 0%
 */

import React from 'react'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { formatNs, formatCount, formatRatio } from '../ui/chart/format'

export interface KpiTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function KpiTile(props: KpiTileProps): JSX.Element {
  const { tile, context, density } = props

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'kpi' }, filter)

  if (query.status === 'loading') {
    return <div className="tile-content">加载中...</div>
  }

  if (query.status === 'error') {
    return <div className="tile-content error">错误: {query.error}</div>
  }

  if (query.status === 'cancelled') {
    return <div className="tile-content">已取消</div>
  }

  const data = query.data
  if (!data) return <div className="tile-content">无数据</div>

  // Compact 模式下只显示关键指标
  if (density === 'compact' || density === 'thumbnail') {
    return (
      <div className="tile-content kpi-compact">
        <div className="kpi-item">
          <div className="label">编译</div>
          <div className="value">{data.compileNs ? formatNs(data.compileNs) : '—'}</div>
        </div>
        <div className="kpi-item">
          <div className="label">Pass</div>
          <div className="value">{formatCount(data.passCount)}</div>
        </div>
        <div className="kpi-item">
          <div className="label">命中率</div>
          <div className="value">{data.cacheHitRate !== null ? formatRatio(data.cacheHitRate) : '—'}</div>
        </div>
      </div>
    )
  }

  // Full / Normal 模式下显示全部指标
  const hitRate = data.cacheHitRate !== null ? formatRatio(data.cacheHitRate) : '—'
  const totalHits = data.cacheExactHit + data.cacheFuzzyHit
  const totalMisses = data.cacheMiss

  return (
    <div className="tile-content kpi-full">
      <div className="kpi-grid">
        <div className="kpi-item">
          <div className="label">编译总耗时</div>
          <div className="value">{data.compileNs ? formatNs(data.compileNs) : '无 compile_module 事件'}</div>
        </div>

        <div className="kpi-item">
          <div className="label">Pass 总耗时</div>
          <div className="value">{formatNs(data.passTotalNs)}</div>
        </div>

        <div className="kpi-item">
          <div className="label">Pass 数量</div>
          <div className="value">{formatCount(data.passCount)}</div>
        </div>

        <div className="kpi-item">
          <div className="label">运行时总耗时</div>
          <div className="value">{data.runtimeNs ? formatNs(data.runtimeNs) : '无 runtime_session_run 事件'}</div>
        </div>

        <div className="kpi-item">
          <div className="label">运行次数</div>
          <div className="value">{formatCount(data.runCount)}</div>
        </div>

        <div className="kpi-item">
          <div className="label">缓存命中率</div>
          <div className="value">{hitRate}</div>
          <div className="note">精确: {formatCount(data.cacheExactHit)}, 模糊: {formatCount(data.cacheFuzzyHit)}, 未命中: {formatCount(data.cacheMiss)}</div>
        </div>

        <div className="kpi-item error">
          <div className="label">错误数</div>
          <div className="value">{formatCount(data.errorCount)}</div>
        </div>

        <div className="kpi-item warn">
          <div className="label">警告数</div>
          <div className="value">{formatCount(data.warnCount)}</div>
        </div>
      </div>

      {/* 无障碍摘要 */}
      <div className="sr-only">
        编译总耗时 {data.compileNs ? formatNs(data.compileNs) : '未记录'}，
        包含 {formatCount(data.passCount)} 个 Pass，
        总耗时 {formatNs(data.passTotalNs)}，
        缓存命中率 {hitRate}，
        错误 {formatCount(data.errorCount)} 个，
        警告 {formatCount(data.warnCount)} 个
      </div>
    </div>
  )
}
