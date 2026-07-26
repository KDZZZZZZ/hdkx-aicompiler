import './tiles.css'
/**
 * IR Before/After Diff Tile：对照某个 Pass 前后的 IR 文本。
 *
 * 查询：用 {kind:'artifact', path} 取 IR 文本
 * irArtifactPath() 拼路径，不要引入 monaco。
 * 纯文本 side-by-side 行对比，自己实现简单 LCS 行 diff，高亮增删行。
 *
 * readiness 为 partial：artifact 存在时可用；从 URL 加载的 bundle 需按需拉取。
 * 重型 Tile（heavy: true）。
 */

import React, { useMemo } from 'react'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { irArtifactPath } from '../kxc/contract'

export interface IrDiffTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

/** 简化的行级 diff：0=unchanged, 1=added, -1=removed */
function simpleLcsDiff(before: string[], after: string[]): Array<{ line: string; status: -1 | 0 | 1 }> {
  const result: Array<{ line: string; status: -1 | 0 | 1 }> = []

  // 极简 LCS：只标记最后一批删除和最后一批添加
  let i = 0
  let j = 0
  while (i < before.length && j < after.length) {
    if (before[i] === after[j]) {
      result.push({ line: before[i] ?? '', status: 0 })
      i++
      j++
    } else {
      break
    }
  }

  // 剩余部分作为删除
  while (i < before.length) {
    result.push({ line: before[i] ?? '', status: -1 })
    i++
  }

  // 剩余部分作为添加
  while (j < after.length) {
    result.push({ line: after[j] ?? '', status: 1 })
    j++
  }

  return result
}

export function IrDiffTile(props: IrDiffTileProps): JSX.Element {
  const { tile, context, density } = props

  if (!context.pass || !context.runId) {
    return <div className="tile-content">请先选择一个 Pass 查看 IR diff</div>
  }

  // 确定 stage（relay/tir/lower）
  const stage: 'relay' | 'tir' | 'lower' = context.pass.includes('lower') ? 'lower' : 'relay'

  const beforePath = irArtifactPath(context.runId ?? '', stage, context.pass ?? '', 'before')
  const afterPath = irArtifactPath(context.runId ?? '', stage, context.pass ?? '', 'after')

  const filter = toQueryFilter(context)
  const beforeQuery = useQuery(context.bundleId, { kind: 'artifact', path: beforePath }, filter)
  const afterQuery = useQuery(context.bundleId, { kind: 'artifact', path: afterPath }, filter)

  const diff = useMemo(() => {
    if (!beforeQuery.data || !afterQuery.data) return null
    const before = beforeQuery.data.content?.split('\n') ?? []
    const after = afterQuery.data.content?.split('\n') ?? []
    return simpleLcsDiff(before, after)
  }, [beforeQuery.data, afterQuery.data])

  if (beforeQuery.status === 'loading' || afterQuery.status === 'loading') {
    return <div className="tile-content">加载 IR artifact...</div>
  }

  if (beforeQuery.status === 'error' || afterQuery.status === 'error') {
    return (
      <div className="tile-content error">
        未找到 artifact
        <p className="note">尝试访问：{beforePath}</p>
      </div>
    )
  }

  if (!diff) {
    return <div className="tile-content">无 IR 数据</div>
  }

  const displayLines = density === 'thumbnail' ? diff.slice(0, 20) : diff.slice(0, 200)

  return (
    <div className="tile-content ir-diff">
      <div className="ir-diff-viewer">
        {displayLines.map((item, idx) => (
          <div key={idx} className={`ir-line status-${item.status}`}>
            <span className="line-num">{idx + 1}</span>
            <span className="line-marker">{item.status === 0 ? ' ' : item.status > 0 ? '+' : '−'}</span>
            <code>{item.line}</code>
          </div>
        ))}
      </div>

      {diff.length > displayLines.length && <div className="note">... 还有 {diff.length - displayLines.length} 行</div>}

      <div className="sr-only">
        IR diff for pass {context.pass}。
        {diff.filter((l) => l.status > 0).length} 行添加，
        {diff.filter((l) => l.status < 0).length} 行删除，
        {diff.filter((l) => l.status === 0).length} 行不变。
      </div>
    </div>
  )
}
