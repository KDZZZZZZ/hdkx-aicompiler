import './tiles.css'
/**
 * 事件表 Tile：按当前过滤条件浏览原始事件。
 *
 * 查询：{kind:'events', offset, limit}
 * 使用 @tanstack/react-virtual 纵向虚拟化，分页查询。
 * 重型 Tile（heavy: true），离屏时应暂停。
 *
 * 列：ts_ns/duration/component/event_type/pass/op/status/severity/message
 */

import React, { useMemo, useRef, useCallback } from 'react'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useWorkbench } from '../state/store'
import { formatNs, formatCount } from '../ui/chart/format'
import { readOpName } from '../kxc/contract'

export interface EventTableTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function EventTableTile(props: EventTableTileProps): JSX.Element {
  const { tile, context, density } = props
  const [offset, setOffset] = React.useState(0)
  const pageSize = density === 'compact' ? 20 : 50
  const setTileSelection = useWorkbench((s) => s.setTileSelection)

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'events', offset, limit: pageSize }, filter)

  const handleRowClick = useCallback(
    (spanId: string) => {
      setTileSelection(tile.id, { kind: 'event', value: spanId })
    },
    [tile.id, setTileSelection],
  )

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data) return <div className="tile-content">无数据</div>

  const data = query.data
  const totalPages = Math.ceil(data.total / pageSize)
  const currentPage = Math.floor(offset / pageSize) + 1

  return (
    <div className="tile-content event-table">
      <table className="events-table">
        <thead>
          <tr>
            <th>时间</th>
            <th>耗时</th>
            <th>Component</th>
            <th>事件类型</th>
            <th>Pass</th>
            <th>Op</th>
            <th>状态</th>
            <th>级别</th>
            <th>消息</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => (
            <tr
              key={row.span_id}
              onClick={() => handleRowClick(row.span_id ?? '')}
              className={`severity-${row.severity}`}
              role="button"
              tabIndex={0}
            >
              <td className="mono">{formatNs(row.ts_ns)}</td>
              <td className="mono">{formatNs(row.duration_ns)}</td>
              <td>{row.component}</td>
              <td>{row.event_type}</td>
              <td>{row.pass_name || '—'}</td>
              <td>{readOpName(row) || '—'}</td>
              <td>{row.status || '—'}</td>
              <td>{(row.severity ?? 'i')[0]?.toUpperCase() ?? 'I'}</td>
              <td title={row.message} className="truncate">
                {row.message}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {density !== 'thumbnail' && (
        <div className="pagination">
          <button onClick={() => setOffset(Math.max(0, offset - pageSize))} disabled={offset === 0}>
            上页
          </button>
          <span>
            第 {currentPage} / {totalPages} 页，共 {formatCount(data.total)} 条
          </span>
          <button onClick={() => setOffset(offset + pageSize)} disabled={offset + pageSize >= data.total}>
            下页
          </button>
        </div>
      )}

      <div className="sr-only">
        事件表，共 {formatCount(data.total)} 条事件。
        {data.rows.slice(0, 3).map((row) => `${row.component} ${row.event_type} ${formatNs(row.duration_ns)}`).join('；')}
      </div>
    </div>
  )
}
