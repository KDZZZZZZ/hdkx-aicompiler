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
 * 把 option 里的 'var(--xxx)' 字符串解析成实际色值。
 *
 * ECharts 画在 canvas 上，不经过 CSS 层，`var(--series-1)` 这种字符串
 * 它并不认识——各 Tile 里这么写的颜色实际上全部失效，只是被 dark 主题的
 * 默认色板兜住了才没露馅。统一在 setOption 前解析一遍，图表代码可以
 * 继续写语义化的变量名（§20 不许硬编码颜色的要求仍然成立）。
 */
function resolveCssVars<T>(value: T, css: CSSStyleDeclaration): T {
  if (typeof value === 'string') {
    const m = /^var\((--[\w-]+)\)$/.exec(value)
    if (m && m[1]) {
      const resolved = css.getPropertyValue(m[1]).trim()
      return (resolved || value) as unknown as T
    }
    return value
  }
  if (Array.isArray(value)) return value.map((v) => resolveCssVars(v, css)) as unknown as T
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = resolveCssVars(v, css)
    }
    return out as unknown as T
  }
  return value
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

  // （颜色解析统一在 resolveCssVars 里做，见下）

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

    // 设置 option（先把 var(--xxx) 解析成实际色值，canvas 不认 CSS 变量）
    instanceRef.current.setOption(
      resolveCssVars(option, getComputedStyle(document.documentElement)),
      true,
    )

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
