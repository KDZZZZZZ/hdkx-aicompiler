/**
 * Gather 模式视图（需求 §4.2）。
 *
 * Gather 是将分散 Tile 临时聚合到一个屏幕上的模式。核心语义：
 * - Tile 是引用视图，不是副本。改变 Gather 中 Tile 的状态会联动原始位置。
 * - 关闭 Gather 中的 Tile 只能调 removeTileFromGather，绝不能调 removeTile（§4.2.5）。
 * - 支持手工拖动调整布局，并在此后显示"已手工调整"标记 + 恢复按钮。
 * - 重型 Tile（heavy=true）同屏限流：最多 3 个完整渲染，其余展示占位（§18.4）。
 *
 * 布局来源：
 * - 如果 activeGatherLayoutId 指向一个有效 layout → 用其 slots
 * - 否则 → 使用 autoGatherLayout() 生成
 * - 手工调整后 auto=false，显示恢复按钮；恢复时重新跑 autoGatherLayout()
 */

import { useState, useCallback, useRef, useMemo } from 'react'
import { useWorkbench } from '../../state/store'
import { getTileSpec } from '../../tiles/registry'
import { autoGatherLayout, groupSlots, gatherColumnCount } from '../../state/gather-layout'
import type { GatherSlot, WidthTier } from '../../state/types'
import { TileFrame } from '../column/TileFrame'
import './modes.css'

/**
 * 重型 Tile 的限流管理：同屏最多 3 个完整活动实例。
 * 不活跃的渲染为静态占位，点击时激活（置换出最久未活动的）。
 */
interface HeavyTilePool {
  /** 当前活跃的重型 tile ID（最多 3 个） */
  activeTileIds: Set<string>
  /** tile ID -> 最后活跃时间戳 */
  lastActivityTime: Map<string, number>
}

export function GatherView(): JSX.Element {
  const workspace = useWorkbench((s) => s.activeWorkspace())
  const gatherSelection = useWorkbench((s) => s.gatherSelection)
  const doc = useWorkbench((s) => s.doc)
  const toggleGatherSelection = useWorkbench((s) => s.toggleGatherSelection)
  const clearGatherSelection = useWorkbench((s) => s.clearGatherSelection)
  const enterGather = useWorkbench((s) => s.enterGather)
  const exitGather = useWorkbench((s) => s.exitGather)
  const removeTileFromGather = useWorkbench((s) => s.removeTileFromGather)
  const updateGatherSlots = useWorkbench((s) => s.updateGatherSlots)
  const resetGatherLayout = useWorkbench((s) => s.resetGatherLayout)
  const saveGatherLayout = useWorkbench((s) => s.saveGatherLayout)
  const activateGatherLayout = useWorkbench((s) => s.activateGatherLayout)
  const deleteGatherLayout = useWorkbench((s) => s.deleteGatherLayout)

  if (!workspace) {
    return <div className="gather-view gather-empty">未选中 Workspace</div>
  }

  // 取当前激活的 Gather 布局
  const activeLayout = workspace.activeGatherLayoutId
    ? workspace.gatherLayouts.find((l) => l.id === workspace.activeGatherLayoutId)
    : null

  // 如果没有激活布局，显示选择界面
  if (!activeLayout) {
    return (
      <div className="gather-view gather-empty">
        <div className="gather-placeholder">
          <h2>进入 Gather 模式</h2>
          <p>从 Strip 中选择多个 Tile，然后点击"进入聚合"或按 G</p>
          <button onClick={() => enterGather()}>进入聚合</button>
        </div>
      </div>
    )
  }

  // 渲染激活的布局
  return (
    <GatherLayoutRenderer
      layout={activeLayout}
      workspace={workspace}
      doc={doc}
      activateGatherLayout={activateGatherLayout}
    />
  )
}

interface GatherLayoutRendererProps {
  layout: any
  workspace: any
  doc: any
  activateGatherLayout: (id: string) => void
}

function GatherLayoutRenderer({
  layout,
  workspace,
  doc,
  activateGatherLayout,
}: GatherLayoutRendererProps): JSX.Element {
  const removeTileFromGather = useWorkbench((s) => s.removeTileFromGather)
  const updateGatherSlots = useWorkbench((s) => s.updateGatherSlots)
  const resetGatherLayout = useWorkbench((s) => s.resetGatherLayout)
  const saveGatherLayout = useWorkbench((s) => s.saveGatherLayout)
  const deleteGatherLayout = useWorkbench((s) => s.deleteGatherLayout)
  const exitGather = useWorkbench((s) => s.exitGather)
  const gatherToWorkspace = useWorkbench((s) => s.gatherToWorkspace)

  // 重型 Tile 限流状态
  const [heavyPool, setHeavyPool] = useState<HeavyTilePool>(() => ({
    activeTileIds: new Set(),
    lastActivityTime: new Map(),
  }))

  const [dragState, setDragState] = useState<{
    draggedTileId?: string
    sourceCol?: number
    sourceRow?: number
  } | null>(null)

  const [saveName, setSaveName] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)

  const containerRef = useRef<HTMLDivElement>(null)

  // 分组 slots（同一格多个 slot → Tab 组）
  const slotGroups = useMemo(() => groupSlots(layout.slots), [layout.slots])
  const colCount = useMemo(() => gatherColumnCount(layout.slots), [layout.slots])

  // 激活重型 Tile（限流到 3 个）
  const activateHeavyTile = useCallback(
    (tileId: string) => {
      setHeavyPool((prev) => {
        const next = { ...prev }
        if (next.activeTileIds.has(tileId)) {
          // 更新活跃时间
          next.lastActivityTime.set(tileId, Date.now())
          return next
        }

        // 如果已经有 3 个了，替换最久未活跃的
        if (next.activeTileIds.size >= 3) {
          let oldest = ''
          let oldestTime = Infinity
          for (const id of next.activeTileIds) {
            const t = next.lastActivityTime.get(id) ?? 0
            if (t < oldestTime) {
              oldestTime = t
              oldest = id
            }
          }
          if (oldest) {
            next.activeTileIds.delete(oldest)
          }
        }

        next.activeTileIds.add(tileId)
        next.lastActivityTime.set(tileId, Date.now())
        return next
      })
    },
    []
  )

  // 处理拖动开始
  const handleDragStart = (e: React.DragEvent, tileId: string, col: number, row: number) => {
    e.dataTransfer.effectAllowed = 'move'
    setDragState({ draggedTileId: tileId, sourceCol: col, sourceRow: row })
  }

  // 处理拖动结束（交换位置）
  const handleDragEnd = useCallback(
    (e: React.DragEvent, targetCol: number, targetRow: number) => {
      if (!dragState?.draggedTileId) return

      const sourceKey = `${dragState.sourceCol}:${dragState.sourceRow}`
      const targetKey = `${targetCol}:${targetRow}`

      if (sourceKey === targetKey) {
        setDragState(null)
        return
      }

      // 交换两个位置的所有 tile。
      // 必须 map 出**新的 slot 对象**：浅拷贝数组后就地改 col/row，改的仍是
      // 当前 doc 里的对象——commit() 克隆的 prev 已被污染，Undo 快照里
      // 存的就是换位后的坐标，撤销等于没撤（Codex review 抓到）。
      const newSlots = layout.slots.map((s: GatherSlot) => {
        const key = `${s.col}:${s.row}`
        if (key === sourceKey) return { ...s, col: targetCol, row: targetRow }
        if (key === targetKey)
          return { ...s, col: dragState.sourceCol ?? 0, row: dragState.sourceRow ?? 0 }
        return s
      })

      // updateGatherSlots 会记录 commit 并置 auto=false
      updateGatherSlots(newSlots)

      setDragState(null)
    },
    [dragState, layout, updateGatherSlots]
  )

  const handleSaveLayout = () => {
    if (saveName.trim()) {
      saveGatherLayout(saveName)
      setSaveName('')
      setShowSaveDialog(false)
    }
  }

  // 计算列宽
  const colWidths: Record<number, WidthTier> = {}
  layout.slots.forEach((s: GatherSlot) => {
    if (!colWidths[s.col]) colWidths[s.col] = s.width
  })

  return (
    <div className="gather-view">
      {/* 工具栏 */}
      <div className="gather-toolbar">
        <div className="gather-toolbar-left">
          <button
            onClick={resetGatherLayout}
            disabled={layout.auto}
            title={layout.auto ? '当前为自动布局' : '恢复自动布局'}
          >
            ↻ 恢复
          </button>
          {!layout.auto && <span className="gather-manual-mark">已手工调整</span>}
          <button onClick={() => setShowSaveDialog(true)}>💾 保存布局</button>

          {/* 已保存布局列表 */}
          {workspace.gatherLayouts.length > 0 && (
            <select
              onChange={(e) => {
                if (e.target.value) {
                  activateGatherLayout(e.target.value)
                }
              }}
              defaultValue={workspace.activeGatherLayoutId || ''}
            >
              <option value="">选择已保存布局</option>
              {workspace.gatherLayouts.map((l: any) => (
                <option key={l.id} value={l.id}>
                  {l.name}
                </option>
              ))}
            </select>
          )}

          {workspace.gatherLayouts.length > 0 && (
            <button
              onClick={() => {
                const id = workspace.activeGatherLayoutId
                if (id) deleteGatherLayout(id)
              }}
            >
              🗑 删除
            </button>
          )}
        </div>

        <div className="gather-toolbar-right">
          <button onClick={() => gatherToWorkspace()}>📋 复制为新 Workspace</button>
          <button onClick={exitGather}>✕ 退出聚合</button>
        </div>
      </div>

      {/* 保存布局对话框 */}
      {showSaveDialog && (
        <div className="gather-dialog-overlay">
          <div className="gather-dialog">
            <h3>保存布局</h3>
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="输入布局名称"
              autoFocus
            />
            <div className="gather-dialog-buttons">
              <button onClick={handleSaveLayout}>保存</button>
              <button onClick={() => setShowSaveDialog(false)}>取消</button>
            </div>
          </div>
        </div>
      )}

      {/* Gather 布局网格 */}
      <div className="gather-grid" ref={containerRef}>
        {Array.from({ length: colCount }).map((_, colIdx) => {
          const width = colWidths[colIdx] ?? '1/3'
          const widthClass =
            width === 'full'
              ? 'w-full'
              : width === '2/3'
                ? 'w-2-3'
                : width === '1/2'
                  ? 'w-1-2'
                  : 'w-1-3'

          return (
            <div
              key={`col-${colIdx}`}
              className={`gather-column ${widthClass}`}
              style={{ '--gather-width': width } as React.CSSProperties}
            >
              {Array.from({ length: 4 }).map((_, rowIdx) => {
                const cellKey = `${colIdx}:${rowIdx}`
                const cellSlots = slotGroups.get(cellKey)

                if (!cellSlots) return null

                // 取该格的首个 Tile 作为代表渲染
                const firstSlot = cellSlots[0]!
                const tile = doc.tiles[firstSlot.tileId]

                if (!tile) return null

                const spec = getTileSpec(tile.type)
                const isHeavy = spec.heavy
                const isActive = heavyPool.activeTileIds.has(firstSlot.tileId)

                // 重型 Tile 限流：只有活跃的才完整渲染
                const shouldRenderFull = !isHeavy || isActive

                return (
                  <div
                    key={cellKey}
                    className={`gather-cell ${cellSlots.length > 1 ? 'has-tabs' : ''}`}
                    draggable={true}
                    onDragStart={(e) => handleDragStart(e, firstSlot.tileId, colIdx, rowIdx)}
                    onDrop={(e) => handleDragEnd(e, colIdx, rowIdx)}
                    onDragOver={(e) => e.preventDefault()}
                  >
                    {shouldRenderFull ? (
                      <>
                        <TileFrame
                          tileId={firstSlot.tileId}
                          density="normal"
                          onRemove={() => removeTileFromGather(firstSlot.tileId)}
                          showSourceBadge={true}
                        />

                        {/* Tab 标签（多个 slot 同一格） */}
                        {cellSlots.length > 1 && (
                          <div className="gather-tabs">
                            {cellSlots.map((slot, idx) => (
                              <div
                                key={idx}
                                className={`gather-tab ${slot.tileId === firstSlot.tileId ? 'active' : ''}`}
                                onClick={() => {
                                  // 简单做法：交换位置使其成为首个
                                  const newSlots = [...layout.slots]
                                  const currentIdx = newSlots.indexOf(slot)
                                  const firstIdx = newSlots.indexOf(firstSlot)
                                  if (currentIdx !== -1 && firstIdx !== -1) {
                                    ;[newSlots[currentIdx], newSlots[firstIdx]] = [
                                      newSlots[firstIdx],
                                      newSlots[currentIdx],
                                    ]
                                    updateGatherSlots(newSlots)
                                  }
                                }}
                              >
                                {doc.tiles[slot.tileId]?.title ||
                                  getTileSpec(doc.tiles[slot.tileId]?.type).title}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      // 重型 Tile 占位（非活跃）
                      <div className="gather-placeholder-heavy">
                        <div className="gather-placeholder-content">
                          <p>{getTileSpec(tile.type).title}</p>
                          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                            已暂停渲染
                          </p>
                          <button
                            onClick={() => activateHeavyTile(firstSlot.tileId)}
                            style={{ marginTop: '8px' }}
                          >
                            点击激活
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}
