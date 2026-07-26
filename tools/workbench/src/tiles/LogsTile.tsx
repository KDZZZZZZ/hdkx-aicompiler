import './tiles.css'
/**
 * 日志 Tile：按级别查看编译期日志与告警。
 *
 * 查询：{kind:'logs', limit}
 * 简化版事件表，只显示日志类事件，支持按 severity 与 component 过滤。
 */

import React, { useMemo } from 'react'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useWorkbench } from '../state/store'
import { formatNs } from '../ui/chart/format'

export interface LogsTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function LogsTile(props: LogsTileProps): JSX.Element {
  const { tile, context, density } = props
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter({ ...context, eventType: 'log' })
  const query = useQuery(context.bundleId, { kind: 'logs', limit: density === 'compact' ? 20 : 50 }, filter)

  const rows = useMemo(() => {
    if (!query.data) return []
    return query.data.rows.slice(0, density === 'compact' ? 10 : 30)
  }, [query.data, density])

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data || rows.length === 0) return <div className="tile-content">无日志</div>

  return (
    <div className="tile-content logs">
      <div className="logs-list">
        {rows.map((row) => (
          <div
            key={row.span_id}
            className={`log-entry severity-${row.severity}`}
            onClick={() => setTileSelection(tile.id, { kind: 'event', value: row.span_id ?? '' })}
            role="button"
            tabIndex={0}
          >
            <span className="severity">{(row.severity ?? 'i')[0]?.toUpperCase() ?? 'I'}</span>
            <span className="component">{row.component}</span>
            <span className="time">{formatNs(row.ts_ns)}</span>
            <span className="message">{row.message}</span>
          </div>
        ))}
      </div>

      <div className="sr-only">
        {query.data.total} 条日志。
        {rows.slice(0, 3).map((row) => `[${row.severity}] ${row.message}`).join('；')}
      </div>
    </div>
  )
}
