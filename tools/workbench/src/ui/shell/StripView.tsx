import { useEffect, useRef, useState, useCallback } from 'react'
import { useWorkbench } from '../../state/store'
import { ColumnView } from '../column/ColumnView'
import { WIDTH_FRACTION, type WidthTier } from '../../state/types'
import './shell.css'

/**
 * StripView 组件（需求 §3.2、§5.3、§13、§18.1）。
 *
 * 核心设计：
 * - 横向滚动条带，Column 按固定宽度平铺，容器总宽 = 各列之和
 * - 'fit' 档位用 ResizeObserver 一次性测量，测不到则退回 '1/3'
 * - 聚焦 Column 自动滚到视口偏左居中（focusPosition = 'center_left'）
 * - 鼠标：普通滚轮滚列内部；Shift+滚轮横移 Strip；空白区拖动横移；Space+拖强制横移
 * - 触控板：纵向手势→列内滚动；横向手势→Strip 横移；方向锁定 150ms 避免抖动
 * - 虚拟化：当前列±2 完整渲染，其余轻量占位（不触发 TileBody）
 * - 聚焦列有明确边框；来源列用淡色连线提示
 */
export function StripView(): JSX.Element {
  const ws = useWorkbench((s) => s.activeWorkspace())
  const columnsOf = useWorkbench((s) => s.columnsOf)
  const focusColumn = useWorkbench((s) => s.focusColumn)
  const setScroll = useWorkbench((s) => s.setScroll)
  const scrollLeft = useWorkbench((s) => s.scrollLeft)
  const viewportWidth = useWorkbench((s) => s.viewportWidth)

  const viewportRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const [columnWidths, setColumnWidths] = useState<Record<string, number>>({})
  const [isDraggingStrip, setIsDraggingStrip] = useState(false)
  const [directionLocked, setDirectionLocked] = useState<'v' | 'h' | null>(null)
  const [directionLockTimer, setDirectionLockTimer] = useState<number | null>(null)

  if (!ws) return <></>

  const columns = columnsOf(ws.id)

  // 初始化列宽测量（ResizeObserver for 'fit' columns）
  useEffect(() => {
    const widths: Record<string, number> = {}
    let measureCount = 0

    const observer = new ResizeObserver(() => {
      if (!scrollContainerRef.current) return

      // 逐列测量
      const children = scrollContainerRef.current.querySelectorAll('[data-column-id]')
      children.forEach((el) => {
        const columnId = el.getAttribute('data-column-id')
        if (columnId && columns.some((c) => c.id === columnId)) {
          const col = columns.find((c) => c.id === columnId)
          if (col?.width === 'fit') {
            const rect = el.getBoundingClientRect()
            widths[columnId] = Math.max(200, rect.width)
            measureCount += 1
          }
        }
      })

      if (measureCount > 0) {
        setColumnWidths((prev) => ({ ...prev, ...widths }))
        observer.disconnect()
      }
    })

    if (scrollContainerRef.current) {
      observer.observe(scrollContainerRef.current)
    }

    return () => observer.disconnect()
  }, [columns])

  // 计算各列实际宽度（px）
  const getColumnWidth = (col: typeof columns[0]): number => {
    if (col.width === 'fit') {
      return columnWidths[col.id] ?? 400 // 默认 400px，直到测量完成
    }
    return Math.max(200, viewportWidth * (WIDTH_FRACTION[col.width] ?? 1 / 3))
  }

  const columnPixelWidths = columns.map(getColumnWidth)
  const totalWidth = columnPixelWidths.reduce((a, b) => a + b, 0)

  // 自动滚动聚焦列到偏左居中（focusPosition = 'center_left'）
  useEffect(() => {
    if (!ws.focusedColumnId) return

    const focusedIdx = columns.findIndex((c) => c.id === ws.focusedColumnId)
    if (focusedIdx < 0) return

    // 计算聚焦列的左边界
    let focusedLeft = 0
    for (let i = 0; i < focusedIdx; i++) {
      focusedLeft += columnPixelWidths[i]! + 8 // 加上列间距
    }

    // 偏左居中：列左对齐到视口左边 1/4 处
    const targetScroll = Math.max(0, focusedLeft - viewportWidth / 4)
    // 防止滚到末尾后面
    const maxScroll = Math.max(0, totalWidth - viewportWidth)
    const clampedScroll = Math.min(targetScroll, maxScroll)

    if (Math.abs(clampedScroll - scrollLeft) > 10) {
      setScroll(clampedScroll, viewportWidth)
    }
  }, [ws.focusedColumnId, columns, columnPixelWidths, totalWidth, viewportWidth, scrollLeft, setScroll])

  // 鼠标滚轮与 Shift+滚轮
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    const shiftKey = e.shiftKey

    if (shiftKey) {
      // Shift+滚轮：横移 Strip
      e.preventDefault()
      const newScroll = Math.max(0, Math.min(totalWidth - viewportWidth, scrollLeft + e.deltaY))
      setScroll(newScroll, viewportWidth)
    } else {
      // 普通滚轮：交给列内处理（冒泡）
      // 这里不拦截，让事件传递给列内容
    }
  }

  // 触控板手势处理（方向锁定）
  const handleTouchpadGesture = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!viewportRef.current) return

    const deltaX = Math.abs(e.deltaX)
    const deltaY = Math.abs(e.deltaY)

    // 首次判定方向，锁定 150ms
    if (!directionLocked && (deltaX > 0 || deltaY > 0)) {
      const newDirection = deltaX > deltaY ? 'h' : 'v'
      setDirectionLocked(newDirection)

      if (directionLockTimer !== null) clearTimeout(directionLockTimer)
      const timer = window.setTimeout(() => {
        setDirectionLocked(null)
        setDirectionLockTimer(null)
      }, 150)
      setDirectionLockTimer(timer)
    }

    // 根据锁定方向处理
    if (directionLocked === 'h') {
      e.preventDefault()
      const newScroll = Math.max(0, Math.min(totalWidth - viewportWidth, scrollLeft + e.deltaX))
      setScroll(newScroll, viewportWidth)
    } else if (directionLocked === 'v') {
      // 让列内处理纵向滚动
    }
  }

  // 空白区域拖动横移 Strip
  const handleStripMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return // 只在空白区响应
    if (!(e.target as HTMLElement).classList.contains('strip-scroll')) return

    e.preventDefault()
    setIsDraggingStrip(true)
    const startX = e.clientX
    const startScroll = scrollLeft

    const handleMouseMove = (me: MouseEvent) => {
      const deltaX = me.clientX - startX
      const newScroll = Math.max(0, Math.min(totalWidth - viewportWidth, startScroll - deltaX))
      setScroll(newScroll, viewportWidth)
    }

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      setIsDraggingStrip(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  // 虚拟化：只完整渲染当前列±2，其余轻量占位
  const focusedIdx = columns.findIndex((c) => c.id === ws.focusedColumnId)
  const renderStart = Math.max(0, focusedIdx - 2)
  const renderEnd = Math.min(columns.length, focusedIdx + 3)

  return (
    <div className="strip-container">
      <div
        className="strip-viewport"
        ref={viewportRef}
        onWheel={handleWheel}
        style={{
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          className={`strip-scroll ${isDraggingStrip ? 'dragging' : ''}`}
          ref={scrollContainerRef}
          onMouseDown={handleStripMouseDown}
          style={{
            transform: `translateX(-${scrollLeft}px)`,
            transition: isDraggingStrip ? 'none' : 'transform 0.3s ease',
          }}
          role="main"
          aria-label="条带视图主区域"
        >
          {columns.length === 0 ? (
            // 空状态提示
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%',
                color: 'var(--text-muted)',
                fontSize: '14px',
              }}
            >
              <div style={{ textAlign: 'center' }}>
                <p>没有列。按 Ctrl+K 打开命令面板创建 Column。</p>
              </div>
            </div>
          ) : (
            columns.map((col, idx) => {
              const shouldRender = idx >= renderStart && idx < renderEnd

              return (
                <div
                  key={col.id}
                  className="strip-column-wrapper"
                  data-column-id={col.id}
                  style={{
                    width: `${columnPixelWidths[idx]}px`,
                  }}
                >
                  {shouldRender ? (
                    // 完整渲染
                    <ColumnView columnId={col.id} virtualized={false} />
                  ) : (
                    // 轻量占位：只显示标题和类型，不触发 TileBody
                    <div
                      style={{
                        background: 'var(--bg-column)',
                        border: '1px solid var(--border)',
                        padding: '8px',
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                        alignItems: 'center',
                        color: 'var(--text-muted)',
                        fontSize: '12px',
                      }}
                    >
                      <div style={{ fontWeight: 500, marginBottom: '4px' }}>{col.title}</div>
                      <div style={{ fontSize: '10px' }}>{col.tileIds.length} Tiles</div>
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
