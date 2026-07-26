/**
 * Focus 模式视图（需求 §4.4）。
 *
 * Focus 让单个 Tile 或 Column 占据主要视口。核心需求：
 * - 读 focusTarget（tile 或 column） 和 focusReturnMode
 * - 保留上下文面包屑：Workspace › Column › Tile，显示当前过滤器摘要
 * - 快速返回原位置（Esc 或退出按钮调 exitFocus()）
 * - 左右方向切换相邻 Tile（同列内）、切换同列 Tab
 * - 保留联动与筛选（不清 context）
 * - 聚焦重型视图时隐藏非必要 UI，给图表让出空间
 */

import { useState, useCallback, useEffect } from 'react'
import { useWorkbench } from '../../state/store'
import { getTileSpec } from '../../tiles/registry'
import type { Tile, Column } from '../../state/types'
import { TileFrame } from '../column/TileFrame'
import './modes.css'

export function FocusView(): JSX.Element {
  const focusTarget = useWorkbench((s) => s.focusTarget)
  const focusReturnMode = useWorkbench((s) => s.focusReturnMode)
  const exitFocus = useWorkbench((s) => s.exitFocus)
  const doc = useWorkbench((s) => s.doc)
  const workspace = useWorkbench((s) => s.activeWorkspace())
  const setActiveTile = useWorkbench((s) => s.setActiveTile)

  if (!focusTarget || !workspace) {
    return <div className="focus-view focus-empty">未选择聚焦目标</div>
  }

  if (focusTarget.kind === 'tile') {
    return <FocusTileView tileId={focusTarget.id} onExit={exitFocus} />
  }

  return <FocusColumnView columnId={focusTarget.id} onExit={exitFocus} />
}

interface FocusTileViewProps {
  tileId: string
  onExit: () => void
}

function FocusTileView({ tileId, onExit }: FocusTileViewProps): JSX.Element {
  const doc = useWorkbench((s) => s.doc)
  const workspace = useWorkbench((s) => s.activeWorkspace())
  const setActiveTile = useWorkbench((s) => s.setActiveTile)

  const tile = doc.tiles[tileId]
  if (!tile || !workspace) return <></>

  const spec = getTileSpec(tile.type)
  const isHeavyTile = spec.heavy

  // 找到该 Tile 所在的 Column
  const columnEntry = Object.entries(doc.columns).find(([, col]) => col.tileIds.includes(tileId))
  const columnId = columnEntry?.[0]
  const column = columnId ? doc.columns[columnId] : null

  if (!column || !columnId) return <></>

  // 获取同列的相邻 Tile
  const tileIndex = column.tileIds.indexOf(tileId)
  const prevTileId = tileIndex > 0 ? column.tileIds[tileIndex - 1] : undefined
  const nextTileId = tileIndex < column.tileIds.length - 1 ? column.tileIds[tileIndex + 1] : undefined
  const prevTile = prevTileId ? doc.tiles[prevTileId] : null
  const nextTile = nextTileId ? doc.tiles[nextTileId] : null

  // 键盘导航
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onExit()
      } else if (e.key === 'ArrowLeft' && prevTile && prevTileId) {
        setActiveTile(columnId, prevTileId)
      } else if (e.key === 'ArrowRight' && nextTile && nextTileId) {
        setActiveTile(columnId, nextTileId)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [columnId, prevTile, prevTileId, nextTile, nextTileId, setActiveTile, onExit])

  return (
    <div className={`focus-view ${isHeavyTile ? 'focus-heavy' : ''}`}>
      {/* 面包屑与控制栏 */}
      {!isHeavyTile && (
        <div className="focus-header">
          <div className="focus-breadcrumb">
            <span className="breadcrumb-part">{workspace.name}</span>
            <span className="breadcrumb-sep">/</span>
            <span className="breadcrumb-part">{column.title}</span>
            <span className="breadcrumb-sep">/</span>
            <span className="breadcrumb-part focus-current">{tile.title || spec.title}</span>
          </div>

          {/* 过滤器摘要 */}
          {Object.keys(tile.context).length > 0 && (
            <div className="focus-context-summary">
              {Object.entries(tile.context)
                .slice(0, 3)
                .map(([key, value]) => (
                  <span key={key} className="context-tag">
                    {key}: {String(value).slice(0, 20)}...
                  </span>
                ))}
            </div>
          )}

          <div className="focus-controls">
            {prevTile && prevTileId && (
              <button
                onClick={() => setActiveTile(columnId, prevTileId)}
                title="上一个 Tile (←)"
                className="focus-nav-btn"
              >
                ← {prevTile.title || getTileSpec(prevTile.type).title}
              </button>
            )}

            {nextTile && nextTileId && (
              <button
                onClick={() => setActiveTile(columnId, nextTileId)}
                title="下一个 Tile (→)"
                className="focus-nav-btn"
              >
                {nextTile.title || getTileSpec(nextTile.type).title} →
              </button>
            )}

            <button onClick={onExit} title="退出聚焦 (Esc)" className="focus-exit-btn">
              ✕ 退出
            </button>
          </div>
        </div>
      )}

      {/* Tile 全屏渲染（给图表最大空间） */}
      <div className="focus-content">
        <TileFrame
          tileId={tileId}
          density={isHeavyTile ? 'full' : 'normal'}
          showSourceBadge={false}
        />
      </div>

      {/* 重型 Tile 的最小控制栏（顶部右侧） */}
      {isHeavyTile && (
        <div className="focus-minimal-controls">
          <button onClick={onExit} title="退出聚焦 (Esc)" className="focus-exit-btn">
            ✕
          </button>
        </div>
      )}
    </div>
  )
}

interface FocusColumnViewProps {
  columnId: string
  onExit: () => void
}

function FocusColumnView({ columnId, onExit }: FocusColumnViewProps): JSX.Element {
  const doc = useWorkbench((s) => s.doc)
  const workspace = useWorkbench((s) => s.activeWorkspace())
  const setActiveTile = useWorkbench((s) => s.setActiveTile)

  const column = doc.columns[columnId]
  if (!column || !workspace) return <></>

  // 同列的首个 Tile
  const activeTileId = column.activeTileId || column.tileIds[0]
  const activeTile = activeTileId ? doc.tiles[activeTileId] : null

  // 相邻列
  const colIndex = workspace.columnIds.indexOf(columnId)
  const prevColumnId = colIndex > 0 ? workspace.columnIds[colIndex - 1] : null
  const nextColumnId = colIndex < workspace.columnIds.length - 1 ? workspace.columnIds[colIndex + 1] : null

  // 键盘导航
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onExit()
      } else if ((e.key === 'ArrowLeft' || e.key === 'h') && prevColumnId) {
        // 切换到前一列
        // 这里只是退出当前聚焦，由外层处理列间切换
      } else if ((e.key === 'ArrowRight' || e.key === 'l') && nextColumnId) {
        // 切换到后一列
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [prevColumnId, nextColumnId, onExit])

  return (
    <div className="focus-view focus-column-view">
      {/* 面包屑 */}
      <div className="focus-header">
        <div className="focus-breadcrumb">
          <span className="breadcrumb-part">{workspace.name}</span>
          <span className="breadcrumb-sep">/</span>
          <span className="breadcrumb-part focus-current">{column.title}</span>
        </div>

        {/* 过滤器摘要 */}
        {Object.keys(column.context).length > 0 && (
          <div className="focus-context-summary">
            {Object.entries(column.context)
              .slice(0, 3)
              .map(([key, value]) => (
                <span key={key} className="context-tag">
                  {key}: {String(value).slice(0, 20)}...
                </span>
              ))}
          </div>
        )}

        <div className="focus-controls">
          {prevColumnId && doc.columns[prevColumnId] && (
            <button className="focus-nav-btn">← {doc.columns[prevColumnId]!.title}</button>
          )}

          {nextColumnId && doc.columns[nextColumnId] && (
            <button className="focus-nav-btn">{doc.columns[nextColumnId]!.title} →</button>
          )}

          <button onClick={onExit} title="退出聚焦 (Esc)" className="focus-exit-btn">
            ✕ 退出
          </button>
        </div>
      </div>

      {/* Column 的所有 Tile（Stack 或 Tab） */}
      <div className="focus-column-content">
        {column.arrange === 'tabs' ? (
          <>
            {/* Tab 模式：只显示激活的 Tile */}
            {activeTile && (
              <>
                <div className="focus-tabs">
                  {column.tileIds.map((tileId) => {
                    const tile = doc.tiles[tileId]
                    if (!tile) return null

                    return (
                      <button
                        key={tileId}
                        className={`focus-tab ${tileId === activeTileId ? 'active' : ''}`}
                        onClick={() => setActiveTile(columnId, tileId)}
                      >
                        {tile.title || getTileSpec(tile.type).title}
                      </button>
                    )
                  })}
                </div>

                <div className="focus-tab-content">
                  {activeTileId && (
                    <TileFrame tileId={activeTileId} density="normal" showSourceBadge={false} />
                  )}
                </div>
              </>
            )}
          </>
        ) : (
          <>
            {/* Stack 模式：显示所有 Tile */}
            {column.tileIds.map((tileId) => {
              const tile = doc.tiles[tileId]
              if (!tile) return null

              return (
                <div key={tileId} className="focus-stacked-tile">
                  <TileFrame tileId={tileId} density="normal" showSourceBadge={false} />
                </div>
              )
            })}
          </>
        )}
      </div>
    </div>
  )
}
