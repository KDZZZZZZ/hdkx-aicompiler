/**
 * 图表格式化工具。
 * 处理时间、字节、百分比等常见数值的显示。
 */

/** 纳秒转可读时间字符串。 */
export function formatNs(ns: number | null | undefined): string {
  if (ns == null || ns === 0) return '0ns'
  if (!Number.isFinite(ns)) return '∞'

  const abs = Math.abs(ns)
  if (abs >= 1e9) {
    const s = ns / 1e9
    return `${s.toFixed(s > 10 ? 0 : 2)}s`
  }
  if (abs >= 1e6) {
    const ms = ns / 1e6
    return `${ms.toFixed(ms > 100 ? 0 : 2)}ms`
  }
  if (abs >= 1e3) {
    const us = ns / 1e3
    return `${us.toFixed(us > 100 ? 0 : 1)}μs`
  }
  return `${ns.toFixed(0)}ns`
}

/** 字节转可读大小。 */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return '—'
  if (!Number.isFinite(bytes)) return '∞'
  if (bytes === 0) return '0B'

  const abs = Math.abs(bytes)
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = abs
  let unitIdx = 0
  while (size >= 1024 && unitIdx < units.length - 1) {
    size /= 1024
    unitIdx++
  }
  const precision = size < 10 ? 2 : size < 100 ? 1 : 0
  return `${size.toFixed(precision)}${units[unitIdx]}`
}

/** 百分比，分母为 0 时返回 "—"。 */
export function formatPercent(
  numerator: number | null | undefined,
  denominator: number | null | undefined,
): string {
  if (numerator == null || denominator == null || denominator === 0) return '—'
  const pct = (numerator / denominator) * 100
  return `${pct.toFixed(1)}%`
}

/** 百分比（已算好的比例 0-1）。 */
export function formatRatio(ratio: number | null | undefined): string {
  if (ratio == null) return '—'
  return `${(ratio * 100).toFixed(1)}%`
}

/** 数字加千位分隔符。 */
export function formatCount(n: number | null | undefined): string {
  if (n == null) return '—'
  return n.toLocaleString('en-US', { maximumFractionDigits: 0 })
}

/** 时间范围的可读表达。 */
export function formatTimeRange(startNs: number, endNs: number): string {
  const durationNs = endNs - startNs
  return `${formatNs(durationNs)} (${formatNs(startNs)} ~ ${formatNs(endNs)})`
}

/** Delta 的符号化表示（用于比较）。 */
export function formatDelta(baseline: number | null, candidate: number | null): string {
  if (baseline == null || candidate == null) return '—'
  const delta = candidate - baseline
  const sign = delta > 0 ? '+' : delta < 0 ? '−' : '±'
  const magnitude = Math.abs(delta / baseline)
  return `${sign}${magnitude.toFixed(1)}%`
}
