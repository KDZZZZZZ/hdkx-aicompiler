import { useRef, useEffect } from 'react'
import { useWorkbench } from '../../state/store'
import { WIDTH_FRACTION, type WidthTier } from '../../state/types'
import './shell.css'

/**
 * Minimap 组件（需求 §5.4、§4.3）。
 * 展示所有 Column 相对位置、当前视口范围、焦点、Pin 列、Warning 列。
 * 支持点击跳转与拖动导航。
 */
export function Minimap(): JSX.Element {
  const ws = useWorkbench((s) => s.activeWorkspace())
  const columnsOf = useWorkbench((s) => s.columnsOf)
  const focusColumn = useWorkbench((s) => s.focusColumn)
  const scrollLeft = useWorkbench((s) => s.scrollLeft)
  const viewportWidth = useWorkbench((s) => s.viewportWidth)
  const setScroll = useWorkbench((s) => s.setScroll)

  const trackRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)

  if (!ws) return <></>

  const columns = columnsOf(ws.id)
  if (columns.length === 0) return <></>

  // 计算每列的宽度（px）
  const columnWidths: number[] = columns.map((c) => {
    if (c.width === 'fit') {
      // fit 列实际宽度由 StripView 的 ResizeObserver 测量，这里默认 400px
      return 400
    }
    return Math.max(200, viewportWidth * (WIDTH_FRACTION[c.width] ?? 1 / 3))
  })

  const totalWidth = columnWidths.reduce((a, b) => a + b, 0)
  const minimapHeight = 22

  // 点击 minimap 跳转
  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!trackRef.current) return
    const rect = trackRef.current.getBoundingClientRect()
    const relX = e.clientX - rect.left
    const scale = totalWidth / rect.width
    const absoluteX = relX * scale

    // 找出点击对应的列
    let acc = 0
    for (let i = 0; i < columns.length; i++) {
      acc += columnWidths[i]!
      if (absoluteX < acc) {
        focusColumn(columns[i]!.id)
        break
      }
    }
  }

  // Minimap 中的拖动导航
  const handleViewportMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!trackRef.current) return
    e.preventDefault()
    const startX = e.clientX
    const startScroll = scrollLeft

    const handleMouseMove = (me: MouseEvent) => {
      if (!trackRef.current) return
      const rect = trackRef.current.getBoundingClientRect()
      const deltaX = me.clientX - startX
      // minimap 宽度到实际宽度的比例
      const scale = totalWidth / rect.width
      const newScroll = Math.max(0, Math.min(totalWidth - viewportWidth, startScroll - deltaX * scale))
      setScroll(newScroll, viewportWidth)
    }

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  // 计算 minimap 中的列指示器和视口框
  const trackWidth = Math.max(200, viewportWidth * 0.8) // minimap 自己的显示宽度
  const scale = trackWidth / totalWidth

  let columnX = 0
  const columnIndicators = columns.map((c, i) => {
    const x = columnX * scale
    const w = columnWidths[i]! * scale
    columnX += columnWidths[i]!

    const hasWarning = false // TODO: 连接诊断数据
    const isGathered = false // TODO: 检查 Gather 中是否使用

    return (
      <div
        key={c.id}
        className={`minimap-column-indicator ${c.id === ws.focusedColumnId ? 'focused' : ''} ${
          c.pinned ? 'pinned' : ''
        } ${hasWarning ? 'warning' : ''}`}
        style={{
          left: `${x}px`,
          width: `${Math.max(2, w)}px`,
        }}
        title={`${c.title}${c.pinned ? ' (固定)' : ''}`}
      />
    )
  })

  // 视口框位置和宽度
  const viewportX = (scrollLeft / totalWidth) * trackWidth
  const viewportW = (viewportWidth / totalWidth) * trackWidth

  return (
    <div className="minimap">
      <div
        className="minimap-track"
        ref={trackRef}
        onClick={handleTrackClick}
        style={{ width: `${trackWidth}px` }}
      >
        {columnIndicators}
        <div
          className="minimap-viewport"
          ref={viewportRef}
          style={{
            left: `${Math.max(0, viewportX)}px`,
            width: `${Math.max(20, viewportW)}px`,
          }}
          onMouseDown={handleViewportMouseDown}
        />
      </div>
    </div>
  )
}
