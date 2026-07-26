/**
 * 所有 Tile 组件的总入口。
 *
 * TileBody 根据 tile.type 分发到具体实现；未实现的类型显示占位符。
 * 每个 Tile 遵循：
 *   - 通过 useQuery 取数，不自己 fetch bundle
 *   - 支持四种 density（full/normal/compact/thumbnail）
 *   - 处理 Loading/Error/Cancelled/Empty 四种状态
 *   - 显示 readiness 提示与 caveat
 *   - 支持联动（点击 setTileSelection、drill down 等）
 *   - 提供 .sr-only 文本摘要
 */

import { useMemo } from 'react'
import type { TileDensity, AnalysisContext } from '../state/types'
import { useWorkbench } from '../state/store'
import { resolveContext, EMPTY_CONTEXT } from '../state/types'
import { getTileSpec } from './registry'

// 导入具体 Tile 实现
import { KpiTile } from './KpiTile'
import { PhaseBreakdownTile } from './PhaseBreakdownTile'
import { HotspotTile } from './HotspotTile'
import { PassWaterfallTile } from './PassWaterfallTile'
import { PassRankingTile } from './PassRankingTile'
import { ShapeCacheHeatmapTile } from './ShapeCacheHeatmapTile'
import { KernelDurationTile } from './KernelDurationTile'
import { KernelTopNTile } from './KernelTopNTile'
import { DiagnosticsTile } from './DiagnosticsTile'
import { RegressionDeltaTile } from './RegressionDeltaTile'
import { LogsTile } from './LogsTile'
import { EventTableTile } from './EventTableTile'
import { TimelineTile } from './TimelineTile'
import { IrDiffTile } from './IrDiffTile'

export interface TileBodyProps {
  tileId: string
  density: TileDensity
}

/**
 * Tile 主体渲染器。
 *
 * 获取 tile 元数据、全局 context、局部 override，
 * 然后根据 type 分发到具体实现。
 */
export function TileBody(props: TileBodyProps): JSX.Element {
  const { tileId, density } = props

  // doc.tiles 是 Record<string, Tile> 而不是数组，直接按 id 取；
  // 这里刻意订阅到单个 tile，避免任何一个 tile 变化就把所有 TileBody 重渲染。
  const tile = useWorkbench((s) => s.doc.tiles[tileId] ?? null)
  // 订阅到 workspace 本身而不是 globalContext()：后者每次调用都返回新对象，
  // 直接当 selector 会让 zustand 认为状态永远在变，导致无限重渲染。
  const workspace = useWorkbench((s) =>
    s.doc.activeWorkspaceId ? (s.doc.workspaces[s.doc.activeWorkspaceId] ?? null) : null,
  )

  const globalContext = useMemo<AnalysisContext | null>(() => {
    if (!workspace) return null
    return {
      ...EMPTY_CONTEXT,
      ...workspace.filters,
      bundleId: workspace.bundleId,
      baselineBundleId: workspace.baselineBundleId,
      candidateBundleId: workspace.compareEnabled ? workspace.bundleId : null,
    }
  }, [workspace])

  const resolvedContext = useMemo(() => {
    if (!globalContext) return EMPTY_CONTEXT
    if (!tile) return globalContext
    return resolveContext(globalContext, tile.context, tile.contextLocked)
  }, [tile, globalContext])

  if (!tile) {
    return <div className="tile-placeholder">Tile 不存在</div>
  }

  const spec = getTileSpec(tile.type)

  return (
    <div className="tile-body" data-tile-type={tile.type} data-density={density}>
      {/* 显示 readiness 提示 */}
      {spec.readiness !== 'ready' && spec.readinessNote && (
        <div className="readiness-banner" role="status">
          <strong>{spec.readiness === 'blocked' ? '⚠ 不可用' : '⚠ 部分数据'}</strong>
          {': '}
          {spec.readinessNote}
        </div>
      )}

      {/* 分发到具体实现 */}
      {tile.type === 'kpi' && (
        <KpiTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'phase_breakdown' && (
        <PhaseBreakdownTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'hotspot_topn' && (
        <HotspotTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'pass_waterfall' && (
        <PassWaterfallTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'pass_ranking' && (
        <PassRankingTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'shape_cache_heatmap' && (
        <ShapeCacheHeatmapTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'kernel_duration_dist' && (
        <KernelDurationTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'kernel_topn' && (
        <KernelTopNTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'diagnostics' && (
        <DiagnosticsTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'regression_delta' && (
        <RegressionDeltaTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'logs' && (
        <LogsTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'event_table' && (
        <EventTableTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'perfetto_timeline' && (
        <TimelineTile tile={tile} context={resolvedContext} density={density} />
      )}
      {tile.type === 'ir_diff' && (
        <IrDiffTile tile={tile} context={resolvedContext} density={density} />
      )}

      {/* 未实现的类型 */}
      {![
        'kpi',
        'phase_breakdown',
        'hotspot_topn',
        'pass_waterfall',
        'pass_ranking',
        'shape_cache_heatmap',
        'kernel_duration_dist',
        'kernel_topn',
        'diagnostics',
        'regression_delta',
        'logs',
        'event_table',
        'perfetto_timeline',
        'ir_diff',
      ].includes(tile.type) && (
        <div className="tile-placeholder">
          <p>Tile type "{tile.type}" not yet implemented</p>
        </div>
      )}
    </div>
  )
}
