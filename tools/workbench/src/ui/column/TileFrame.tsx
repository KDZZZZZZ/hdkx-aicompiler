import { useState, useRef, type ReactNode } from 'react'
import { useWorkbench } from '../../state/store'
import { getTileSpec } from '../../tiles/registry'
import { TileBody } from '../../tiles/components'
import type { TileDensity } from '../../state/types'
import './column.css'

export interface TileFrameProps {
  tileId: string
  density?: TileDensity
  onRemove?: () => void
  showSourceBadge?: boolean
  headerExtra?: ReactNode
  /** 重型 Tile 被限流暂停时为 true，此时不挂载图表主体（§18.2/§18.4）。 */
  paused?: boolean
  onResume?: () => void
}

/**
 * TileFrame 组件（需求 §6、§20.3）。
 * 标题栏提供：标题、图表类型、数据范围摘要、Bundle 标识、Baseline 状态、
 * Loading、Warning、Refresh、Drill Down、Pin、Add to Gather、Duplicate、Move、
 * Tab/Stack、Focus、More、Close。
 *
 * 紧凑模式下低频操作折叠进 More 菜单。
 * readiness 非 ready 时显示警告标记和 readinessNote。
 * Warning 必须同时有图标或文字，不仅限颜色（§21）。
 */
export function TileFrame(props: TileFrameProps): JSX.Element {
  const {
    tileId,
    density = 'normal',
    onRemove,
    showSourceBadge = false,
    headerExtra,
    paused = false,
    onResume,
  } = props

  const tile = useWorkbench((s) => s.doc.tiles[tileId])
  const removeTile = useWorkbench((s) => s.removeTile)
  const duplicateTile = useWorkbench((s) => s.duplicateTile)
  const enterFocus = useWorkbench((s) => s.enterFocus)

  const [showMoreMenu, setShowMoreMenu] = useState(false)
  const moreMenuRef = useRef<HTMLDivElement>(null)

  if (!tile) return <></>

  const spec = getTileSpec(tile.type)
  const title = tile.title || spec.title
  const isCompact = density === 'compact' || density === 'thumbnail'

  // 警告标记：readiness 非 ready 时显示
  const showWarning = spec.readiness !== 'ready'

  const handleRemove = () => {
    if (onRemove) {
      onRemove()
    } else {
      removeTile(tileId)
    }
  }

  const handleDuplicate = () => {
    duplicateTile(tileId)
  }

  const handleFocus = () => {
    enterFocus('tile', tileId)
  }

  // 点击外部关闭 More 菜单
  const handleDocumentClick = (e: MouseEvent) => {
    if (moreMenuRef.current && !moreMenuRef.current.contains(e.target as Node)) {
      setShowMoreMenu(false)
    }
  }

  return (
    <div className={`tile-frame ${isCompact ? 'compact' : ''}`}>
      {/* 标题栏 */}
      <div className="tile-header">
        <div className="tile-header-start">
          <span className="tile-title" title={title}>
            {title}
          </span>

          {/* 图表类型徽章 */}
          <span className="tile-type-badge">{spec.type}</span>

          {/* readiness 警告标记（§6 必须） */}
          {showWarning && (
            <div className="tile-readiness-warning" title={spec.readinessNote || '数据可用性提示'}>
              <span className="tile-readiness-icon">!</span>
              <span>{spec.readiness === 'partial' ? '部分' : '数据缺失'}</span>
            </div>
          )}

          {/* 来源列标记（Gather 中用） */}
          {showSourceBadge && tile.sourceTileId && (
            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>📌 引用</span>
          )}

          {headerExtra}
        </div>

        {/* 控制按钮 */}
        <div className="tile-header-controls">
          {/* 刷新（暂未连接数据） */}
          <button className="tile-btn" title="刷新" aria-label="刷新图表">
            🔄
          </button>

          {/* Drill Down */}
          <button className="tile-btn" title="深入分析" aria-label="执行下钻">
            ↓
          </button>

          {/* Pin */}
          <button className="tile-btn" title="固定" aria-label="固定此图">
            📌
          </button>

          {/* Add to Gather */}
          <button className="tile-btn" title="加入聚合" aria-label="添加到 Gather">
            ➕
          </button>

          {/* Duplicate */}
          <button className="tile-btn" onClick={handleDuplicate} title="复制" aria-label="复制此图">
            📋
          </button>

          {/* Focus */}
          <button className="tile-btn" onClick={handleFocus} title="全屏查看" aria-label="全屏查看此图">
            ⛶
          </button>

          {/* More 菜单 */}
          <div style={{ position: 'relative' }} ref={moreMenuRef}>
            <button
              className="tile-btn tile-btn-more"
              onClick={() => setShowMoreMenu(!showMoreMenu)}
              title="更多选项"
              aria-label="打开更多选项菜单"
            >
              ⋯
            </button>

            {showMoreMenu && (
              <div
                style={{
                  position: 'absolute',
                  right: 0,
                  top: '100%',
                  background: 'var(--bg-raised)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  minWidth: '120px',
                  zIndex: 1000,
                  marginTop: '4px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                }}
              >
                <button
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: '6px 8px',
                    fontSize: '12px',
                    border: 'none',
                    background: 'transparent',
                    color: 'inherit',
                    cursor: 'pointer',
                  }}
                  onClick={() => {
                    handleDuplicate()
                    setShowMoreMenu(false)
                  }}
                >
                  复制
                </button>
                <button
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: '6px 8px',
                    fontSize: '12px',
                    border: 'none',
                    background: 'transparent',
                    color: 'inherit',
                    cursor: 'pointer',
                  }}
                  onClick={() => {
                    handleFocus()
                    setShowMoreMenu(false)
                  }}
                >
                  全屏
                </button>
                <button
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: '6px 8px',
                    fontSize: '12px',
                    border: 'none',
                    background: 'transparent',
                    color: 'inherit',
                    cursor: 'pointer',
                  }}
                  onClick={() => {
                    handleRemove()
                    setShowMoreMenu(false)
                  }}
                >
                  关闭
                </button>
              </div>
            )}
          </div>

          {/* Close */}
          <button
            className="tile-btn tile-btn-close"
            onClick={handleRemove}
            title="关闭"
            aria-label="关闭此图"
          >
            ✕
          </button>
        </div>
      </div>

      {/* 图表主体。重型 Tile 被暂停时不挂载 TileBody，
          这样 ECharts 实例与查询都不会发生（§18.2 Pause/Dispose）。 */}
      <div className="tile-body-slot">
        {paused ? (
          <button
            className="tile-body-placeholder tile-body-paused"
            onClick={onResume}
            title="该视图较重，已暂停渲染以保证其他图表流畅"
          >
            {spec.title} · 已暂停渲染，点击激活
          </button>
        ) : (
          <TileBody tileId={tileId} density={density} />
        )}
      </div>
    </div>
  )
}
