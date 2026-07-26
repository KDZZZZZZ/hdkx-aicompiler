/**
 * ECharts 实例生命周期管理（需求 §18.2 重型 Tile 的 Lazy Load / Pause / Dispose）。
 *
 * 约束：
 * - option 为 null 时不创建实例，已创建则销毁，便于重型 Tile 暂停渲染。
 * - 容器尺寸变化时自动 resize。
 * - 组件卸载时 dispose。
 * - 从 tokens.css 读取 CSS 变量用于颜色，不要硬编码十六进制。
 */

import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'

export interface UseEChartsOpts {
  onEvents?: Record<string, (params: unknown) => void>
  theme?: 'dark' | 'light'
}

/**
 * ECharts 实例容器管理。
 *
 * 用法：
 *   const containerRef = useECharts(option, { onEvents: { click: handleClick } })
 *   return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
 *
 * 返回的 ref 既是容器、又是管理器，在 option/onEvents 变化时自动更新实例。
 */
export function useECharts(
  option: EChartsOption | null,
  opts?: UseEChartsOpts,
): React.RefObject<HTMLDivElement> {
  const containerRef = useRef<HTMLDivElement>(null)
  const instanceRef = useRef<echarts.ECharts | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  // 从 tokens.css 读取颜色
  const getSeriesColor = (index: number): string => {
    const varName = `--series-${Math.min(index + 1, 8)}`
    const computed = getComputedStyle(document.documentElement)
    return computed.getPropertyValue(varName).trim() || '#5b9dd9'
  }

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // option 为 null 时销毁实例
    if (!option) {
      if (instanceRef.current) {
        instanceRef.current.dispose()
        instanceRef.current = null
      }
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect()
        resizeObserverRef.current = null
      }
      return
    }

    // 创建或获取实例
    if (!instanceRef.current) {
      instanceRef.current = echarts.init(container, opts?.theme || 'dark')
      // 绑定事件
      if (opts?.onEvents) {
        for (const [event, handler] of Object.entries(opts.onEvents)) {
          instanceRef.current.on(event, handler)
        }
      }
    }

    // 设置 option
    instanceRef.current.setOption(option, true)

    // 设置 ResizeObserver 以在容器尺寸变化时 resize
    if (!resizeObserverRef.current && container.parentElement) {
      resizeObserverRef.current = new ResizeObserver(() => {
        if (instanceRef.current && container.offsetWidth > 0) {
          instanceRef.current.resize()
        }
      })
      resizeObserverRef.current.observe(container.parentElement)
    }
  }, [option, opts?.onEvents, opts?.theme])

  // 卸载时清理
  useEffect(() => {
    return () => {
      if (instanceRef.current) {
        instanceRef.current.dispose()
        instanceRef.current = null
      }
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect()
        resizeObserverRef.current = null
      }
    }
  }, [])

  return containerRef
}
