import { useState, useRef } from 'react'
import { useWorkbench } from '../../state/store'
import { TileFrame } from './TileFrame'
import { WIDTH_TIERS, type WidthTier, type TileDensity, type Column } from '../../state/types'
import './column.css'

export interface ColumnViewProps {
  columnId: string
  virtualized?: boolean
}

/**
 * ColumnView 组件（需求 §3.3、§6、§12、§18.1）。
 * 列标题（可双击重命名）、宽度档位切换、Pin 按钮、Stack/Tab 切换、
 * 列内 Tile 纵向平铺或 Tab 展示、Tile 之间可拖动分隔条调高度。
 * Compare 角色标记、来源列连线（简化）。
 */
export function ColumnView(props: ColumnViewProps): JSX.Element {
  const { columnId } = props

  const doc = useWorkbench((s) => s.doc)
  const column = doc.columns[columnId]
  const tilesOf = useWorkbench((s) => s.tilesOf)
  const setColumnWidth = useWorkbench((s) => s.setColumnWidth)
  const setColumnArrange = useWorkbench((s) => s.setColumnArrange)
  const togglePin = useWorkbench((s) => s.togglePin)
  const setActiveTile = useWorkbench((s) => s.setActiveTile)
  const setTileHeight = useWorkbench((s) => s.setTileHeight)
  const focusColumn = useWorkbench((s) => s.focusColumn)
  const removeColumn = useWorkbench((s) => s.removeColumn)
  const ws = useWorkbench((s) => s.activeWorkspace())

  const [isRenamingTitle, setIsRenamingTitle] = useState(false)
  const [editingTitle, setEditingTitle] = useState(column?.title ?? '')
  const [draggingSplitter, setDraggingSplitter] = useState<string | null>(null)
  const titleInputRef = useRef<HTMLInputElement>(null)
  const columnRef = useRef<HTMLDivElement>(null)

  if (!column) return <></>

  const tiles = tilesOf(columnId)
  const isFocused = ws?.focusedColumnId === columnId
  const isCompact = column.width === '1/3'

  // 计算 Tile 高度：均分或自定义
  const getTileHeight = (index: number): number => {
    const tile = tiles[index]
    if (!tile) return 200
    if (tile.heightPx !== null) return tile.heightPx
    // 默认按列宽推荐高度均分
    const defaultHeight = 240
    return defaultHeight
  }

  // 处理标题编辑
  const handleTitleDoubleClick = () => {
    setIsRenamingTitle(true)
    setEditingTitle(column.title)
  }

  const handleTitleSave = () => {
    if (editingTitle && editingTitle !== column.title) {
      // TODO: 调用 renameColumn 或 setColumnTitle
    }
    setIsRenamingTitle(false)
  }

  const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleTitleSave()
    } else if (e.key === 'Escape') {
      setIsRenamingTitle(false)
    }
  }

  // 处理 Tile 高度分隔条拖动
  const handleSplitterMouseDown = (tileId: string, e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDraggingSplitter(tileId)
    const startY = e.clientY
    const tile = doc.tiles[tileId]
    const startHeight = tile?.heightPx ?? getTileHeight(tiles.indexOf(doc.tiles[tileId]!))

    const handleMouseMove = (me: MouseEvent) => {
      const deltaY = me.clientY - startY
      const newHeight = Math.max(100, (startHeight ?? 200) + deltaY)
      setTileHeight(tileId, newHeight)
    }

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      setDraggingSplitter(null)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  // Stack 模式：平铺显示所有 Tile
  if (column.arrange === 'stack') {
    return (
      <div
        ref={columnRef}
        className={`column ${isFocused ? 'focused' : ''}`}
        role="region"
        aria-label={`列: ${column.title}`}
      >
        {/* 列头 */}
        <div className="column-header">
          {isRenamingTitle ? (
            <input
              ref={titleInputRef}
              type="text"
              className="column-title-input"
              value={editingTitle}
              onChange={(e) => setEditingTitle(e.currentTarget.value)}
              onBlur={handleTitleSave}
              onKeyDown={handleTitleKeyDown}
              autoFocus
            />
          ) : (
            <div
              className="column-title editable"
              onDoubleClick={handleTitleDoubleClick}
              onClick={() => focusColumn(columnId)}
              title={`${column.title} - 双击重命名`}
            >
              {column.title}
            </div>
          )}

          <div className="column-controls">
            {/* Compare 标记 */}
            {column.compareRole && (
              <span className={`column-compare-badge ${column.compareRole}`}>
                {column.compareRole === 'baseline' ? 'BL' : 'CD'}
              </span>
            )}

            {/* Stack/Tab 切换 */}
            <select
              className="column-arrange-selector"
              value={column.arrange}
              onChange={(e) => setColumnArrange(columnId, e.currentTarget.value as 'stack' | 'tabs')}
              aria-label="切换排列方式"
            >
              <option value="stack">平铺</option>
              <option value="tabs">标签</option>
            </select>

            {/* 宽度档位 */}
            <select
              className="column-arrange-selector"
              value={column.width}
              onChange={(e) => setColumnWidth(columnId, e.currentTarget.value as WidthTier)}
              aria-label="调整列宽"
            >
              {WIDTH_TIERS.map((w) => (
                <option key={w} value={w}>
                  {w}
                </option>
              ))}
            </select>

            {/* Pin 按钮 */}
            <button
              className="column-btn"
              onClick={() => togglePin(columnId)}
              title={column.pinned ? '取消固定' : '固定此列'}
              aria-label={column.pinned ? '取消固定此列' : '固定此列'}
              style={{ color: column.pinned ? 'var(--pin-accent)' : 'inherit' }}
            >
              📌
            </button>

            {/* 关闭列 */}
            <button
              className="column-btn"
              onClick={() => removeColumn(columnId)}
              title="关闭此列"
              aria-label="关闭此列"
            >
              ✕
            </button>
          </div>
        </div>

        {/* 列体：Tile 平铺 */}
        <div className="column-body stack">
          {tiles.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', padding: '20px', textAlign: 'center' }}>
              此列无 Tile — 按 Ctrl+K 打开命令面板添加
            </div>
          ) : (
            tiles.map((tile, i) => (
              <div key={tile.id}>
                <div
                  style={{
                    height: `${getTileHeight(i)}px`,
                    overflow: 'hidden',
                  }}
                >
                  <TileFrame
                    tileId={tile.id}
                    density={isCompact ? 'compact' : 'normal'}
                    showSourceBadge={false}
                  />
                </div>
                {i < tiles.length - 1 && (
                  <div
                    className={`tile-splitter ${draggingSplitter === tile.id ? 'dragging' : ''}`}
                    onMouseDown={(e) => handleSplitterMouseDown(tile.id, e)}
                  />
                )}
              </div>
            ))
          )}
        </div>
      </div>
    )
  }

  // Tab 模式：只显示活动 Tile
  const activeTile = column.activeTileId ? doc.tiles[column.activeTileId] : null

  return (
    <div
      ref={columnRef}
      className={`column ${isFocused ? 'focused' : ''}`}
      role="region"
      aria-label={`列: ${column.title}`}
    >
      {/* 列头 */}
      <div className="column-header">
        {isRenamingTitle ? (
          <input
            ref={titleInputRef}
            type="text"
            className="column-title-input"
            value={editingTitle}
            onChange={(e) => setEditingTitle(e.currentTarget.value)}
            onBlur={handleTitleSave}
            onKeyDown={handleTitleKeyDown}
            autoFocus
          />
        ) : (
          <div
            className="column-title editable"
            onDoubleClick={handleTitleDoubleClick}
            onClick={() => focusColumn(columnId)}
            title={`${column.title} - 双击重命名`}
          >
            {column.title}
          </div>
        )}

        <div className="column-controls">
          {/* Compare 标记 */}
          {column.compareRole && (
            <span className={`column-compare-badge ${column.compareRole}`}>
              {column.compareRole === 'baseline' ? 'BL' : 'CD'}
            </span>
          )}

          {/* Stack/Tab 切换 */}
          <select
            className="column-arrange-selector"
            value={column.arrange}
            onChange={(e) => setColumnArrange(columnId, e.currentTarget.value as 'stack' | 'tabs')}
            aria-label="切换排列方式"
          >
            <option value="stack">平铺</option>
            <option value="tabs">标签</option>
          </select>

          {/* 宽度档位 */}
          <select
            className="column-arrange-selector"
            value={column.width}
            onChange={(e) => setColumnWidth(columnId, e.currentTarget.value as WidthTier)}
            aria-label="调整列宽"
          >
            {WIDTH_TIERS.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>

          {/* Pin 按钮 */}
          <button
            className="column-btn"
            onClick={() => togglePin(columnId)}
            title={column.pinned ? '取消固定' : '固定此列'}
            aria-label={column.pinned ? '取消固定此列' : '固定此列'}
            style={{ color: column.pinned ? 'var(--pin-accent)' : 'inherit' }}
          >
            📌
          </button>

          {/* 关闭列 */}
          <button
            className="column-btn"
            onClick={() => removeColumn(columnId)}
            title="关闭此列"
            aria-label="关闭此列"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Tab 栏 */}
      <div className="column-tabs">
        {tiles.map((tile) => (
          <button
            key={tile.id}
            className={`column-tab ${tile.id === column.activeTileId ? 'active' : ''}`}
            onClick={() => setActiveTile(columnId, tile.id)}
            title={tile.title || `Tile: ${tile.type}`}
          >
            {tile.title || tile.type}
          </button>
        ))}
      </div>

      {/* Tab 内容 */}
      <div className="column-tab-content">
        {activeTile ? (
          <TileFrame
            tileId={activeTile.id}
            density={isCompact ? 'compact' : 'normal'}
            showSourceBadge={false}
          />
        ) : (
          <div style={{ color: 'var(--text-muted)', padding: '20px', textAlign: 'center' }}>
            此列无 Tile
          </div>
        )}
      </div>
    </div>
  )
}
