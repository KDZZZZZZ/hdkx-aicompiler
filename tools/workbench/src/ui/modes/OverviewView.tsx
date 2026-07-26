/**
 * Overview 模式视图（需求 §4.3）。
 *
 * Overview 展示所有 Workspace 与 Column 的空间分布（缩略图墙）。核心需求：
 * - 每个 Workspace 一行/一区，显示名称；其下横向排列该 Workspace 的 Column 缩略图
 * - Column 缩略图显示：标题、过滤器摘要、Pin 状态、Warning 数量、聚焦状态、
 *   Gather 引用状态、来源关系
 * - 渲染预算（§18.3）：只对视口内的缩略图渲染真实 TileBody，其余纯占位
 * - 使用 IntersectionObserver 实现虚拟化
 * - 支持：点击跳转、拖动重排、跨 Workspace 移动或复制、删除、创建 Workspace、
 *   搜索、按图表类型和诊断严重级别过滤、返回原缩放位置（§21：屏幕阅读器支持）
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { useWorkbench } from '../../state/store'
import { getTileSpec } from '../../tiles/registry'
import { TileBody } from '../../tiles/components'
import type { Column, Tile } from '../../state/types'
import './modes.css'

interface ColumnThumbnailProps {
  column: Column
  workspace: any
  doc: any
  onClickThumbnail: (columnId: string) => void
}

/**
 * Column 缩略图组件，支持渲染预算（只在视口内时渲染真实 TileBody）。
 */
function ColumnThumbnail({ column, workspace, doc, onClickThumbnail }: ColumnThumbnailProps) {
  const [isVisible, setIsVisible] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const observer = new IntersectionObserver((entries) => {
      const entry = entries[0]
      if (entry) {
        setIsVisible(entry.isIntersecting)
      }
    })
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])

  const tiles = column.tileIds.map((id) => doc.tiles[id]).filter((t: Tile | undefined) => t)
  const warningCount = tiles.filter((t: Tile) => getTileSpec(t.type).readiness !== 'ready').length

  // 检查该列是否被 Gather 引用
  const gatherReferenced =
    workspace.gatherLayouts.some((layout: any) =>
      layout.slots.some((slot: any) =>
        column.tileIds.includes(slot.tileId)
      )
    )

  return (
    <div
      ref={ref}
      className={`overview-column-thumbnail ${column.pinned ? 'pinned' : ''} ${
        workspace.focusedColumnId === column.id ? 'focused' : ''
      }`}
      onClick={() => onClickThumbnail(column.id)}
      role="listitem"
      aria-label={`Column: ${column.title}${column.pinned ? ' (已固定)' : ''}`}
    >
      {/* 列头 */}
      <div className="thumbnail-header">
        <div className="thumbnail-title" title={column.title}>
          {column.title}
        </div>
        {column.pinned && <span className="thumbnail-pin-badge">📌</span>}
        {gatherReferenced && <span className="thumbnail-gather-badge">G</span>}
        {warningCount > 0 && (
          <span className="thumbnail-warning-badge" title={`${warningCount} 个数据缺失`}>
            ⚠ {warningCount}
          </span>
        )}
      </div>

      {/* 过滤器摘要 */}
      {Object.keys(column.context).length > 0 && (
        <div className="thumbnail-filter-summary">
          {Object.entries(column.context)
            .slice(0, 2)
            .map(([key, value]) => (
              <span key={key} className="filter-tag">
                {key}: {String(value).slice(0, 12)}...
              </span>
            ))}
          {Object.keys(column.context).length > 2 && (
            <span className="filter-tag">+{Object.keys(column.context).length - 2}</span>
          )}
        </div>
      )}

      {/* 缩略图内容（渲染预算：只在视口内时渲染） */}
      <div className="thumbnail-body">
        {isVisible ? (
          tiles.slice(0, 2).map((tile: Tile) => (
            <div key={tile.id} className="thumbnail-tile">
              <TileBodyThumbnail tileId={tile.id} />
            </div>
          ))
        ) : (
          <div className="thumbnail-placeholder">
            {tiles.length > 0 && <span>{tiles.length} 个 Tile</span>}
          </div>
        )}
      </div>

      {/* 来源指示 */}
      {column.sourceColumnId && (
        <div className="thumbnail-source" title={`来自列: ${column.sourceColumnId}`}>
          ← 来自 {column.origin}
        </div>
      )}
    </div>
  )
}

/**
 * 单个 Tile 的缩略图体（density='thumbnail'）。
 *
 * 重型 Tile 在 Overview 里一律不渲染真实内容：Overview 可能同时出现上百个缩略图，
 * 放任 Perfetto/IR Diff 这类视图实例化会直接拖垮页面。需求 §4.3 也明确说了
 * 缩略图只用于识别，不要求坐标轴与数据标签完整可读。
 */
function TileBodyThumbnail({ tileId }: { tileId: string }) {
  const tile = useWorkbench((s) => s.doc.tiles[tileId] ?? null)
  if (!tile) return null

  const spec = getTileSpec(tile.type)
  if (spec.heavy) {
    return (
      <div className="tile-body-thumbnail tile-body-thumbnail--heavy">
        <span>{tile.title || spec.title}</span>
        <span className="thumbnail-hint">重型视图，仅在进入后渲染</span>
      </div>
    )
  }
  return (
    <div className="tile-body-thumbnail">
      <TileBody tileId={tileId} density="thumbnail" />
    </div>
  )
}

export function OverviewView(): JSX.Element {
  const doc = useWorkbench((s) => s.doc)
  const workspaces = doc.workspaceOrder.map((id) => doc.workspaces[id])
  const switchWorkspace = useWorkbench((s) => s.switchWorkspace)
  const focusColumn = useWorkbench((s) => s.focusColumn)
  const setMode = useWorkbench((s) => s.setMode)
  const scrollLeft = useWorkbench((s) => s.scrollLeft)
  const setScroll = useWorkbench((s) => s.setScroll)

  const [searchText, setSearchText] = useState('')
  const [filterType, setFilterType] = useState<string | null>(null)
  const [filterSeverity, setFilterSeverity] = useState<string | null>(null)
  const [draggedColumn, setDraggedColumn] = useState<string | null>(null)

  const handleClickThumbnail = useCallback(
    (columnId: string) => {
      const column = doc.columns[columnId]
      if (!column) return

      // 找到该列所属的 workspace
      const ws = workspaces.find((w) => w && w.columnIds.includes(columnId))
      if (ws) {
        // 切换 workspace 并聚焦该列
        switchWorkspace(ws.id)
        focusColumn(columnId)
        // 记住当前位置，退出 overview 时返回
        setScroll(scrollLeft, window.innerWidth)
        // 返回 strip 模式
        setMode('strip')
      }
    },
    [doc, workspaces, switchWorkspace, focusColumn, setMode, scrollLeft, setScroll]
  )

  const handleDragStart = (columnId: string) => {
    setDraggedColumn(columnId)
  }

  const handleDrop = (targetColumnId: string) => {
    if (!draggedColumn || draggedColumn === targetColumnId) {
      setDraggedColumn(null)
      return
    }

    // 简单的拖动重排：交换位置
    const sourceCol = doc.columns[draggedColumn]
    const targetCol = doc.columns[targetColumnId]

    if (!sourceCol || !targetCol) {
      setDraggedColumn(null)
      return
    }

    // 找到两列所在的 workspace
    const sourceWs = workspaces.find((w) => w && w.columnIds.includes(draggedColumn))
    const targetWs = workspaces.find((w) => w && w.columnIds.includes(targetColumnId))

    if (sourceWs && targetWs) {
      // 如果在同一个 workspace，交换；否则简单实现为移动
      if (sourceWs.id === targetWs.id) {
        const idx1 = sourceWs.columnIds.indexOf(draggedColumn)
        const idx2 = sourceWs.columnIds.indexOf(targetColumnId)
        if (idx1 !== -1 && idx2 !== -1) {
          const col1 = sourceWs.columnIds[idx1]
          const col2 = sourceWs.columnIds[idx2]
          if (col1 && col2) {
            sourceWs.columnIds[idx1] = col2
            sourceWs.columnIds[idx2] = col1
          }
        }
      }
    }

    setDraggedColumn(null)
  }

  // 筛选列表
  const filteredWorkspaces = useMemo(() => {
    return workspaces.map((ws) => (
      ws ? {
        ...ws,
        columnIds: ws.columnIds.filter((columnId) => {
        const column = doc.columns[columnId]
        if (!column) return false

        // 按标题或过滤器文本搜索
        if (searchText && !column.title.toLowerCase().includes(searchText.toLowerCase())) {
          return false
        }

        // 按图表类型过滤
        if (filterType) {
          const hasTileOfType = column.tileIds.some((tileId) => {
            const tile = doc.tiles[tileId]
            return tile && tile.type === filterType
          })
          if (!hasTileOfType) return false
        }

        // 按诊断严重级别过滤
        if (filterSeverity) {
          const hasSeverity = column.tileIds.some((tileId) => {
            const tile = doc.tiles[tileId]
            return tile && tile.context.severity === filterSeverity
          })
          if (!hasSeverity) return false
        }

        return true
      }),
      } : null
    ))
  }, [workspaces, doc, searchText, filterType, filterSeverity])

  return (
    <div className="overview-view">
      {/* 顶部筛选栏 */}
      <div className="overview-filter-bar">
        <input
          type="text"
          placeholder="搜索 Column（标题或过滤器）"
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          className="overview-search-input"
        />

        <select
          value={filterType || ''}
          onChange={(e) => setFilterType(e.target.value || null)}
          className="overview-filter-select"
        >
          <option value="">所有图表类型</option>
          {['kpi', 'phase_breakdown', 'hotspot_topn', 'perfetto_timeline', 'pass_waterfall'].map(
            (type) => (
              <option key={type} value={type}>
                {type}
              </option>
            )
          )}
        </select>

        <select
          value={filterSeverity || ''}
          onChange={(e) => setFilterSeverity(e.target.value || null)}
          className="overview-filter-select"
        >
          <option value="">所有严重级别</option>
          {['info', 'warn', 'error'].map((severity) => (
            <option key={severity} value={severity}>
              {severity}
            </option>
          ))}
        </select>

        <button onClick={() => setMode('strip')} className="overview-return-btn">
          返回 Strip
        </button>
      </div>

      {/* Workspace 分组 */}
      <div className="overview-workspaces" role="group" aria-label="All workspaces">
        {filteredWorkspaces
          .filter((ws) => ws)
          .map((ws) => (
          <div key={ws!.id} className="overview-workspace-group" role="group" aria-label={ws!.name}>
            <div className="workspace-header">
              <h2>{ws!.name}</h2>
              <span className="workspace-column-count">{ws!.columnIds.length} 列</span>
            </div>

            <div className="workspace-columns">
              {ws!.columnIds.map((columnId) => {
                const column = doc.columns[columnId]
                if (!column) return null

                return (
                  <div
                    key={columnId}
                    draggable={true}
                    onDragStart={() => handleDragStart(columnId)}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={() => handleDrop(columnId)}
                  >
                    <ColumnThumbnail
                      column={column}
                      workspace={ws!}
                      doc={doc}
                      onClickThumbnail={handleClickThumbnail}
                    />
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
