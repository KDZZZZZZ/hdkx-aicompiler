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
import { lcsDiff } from '../ui/chart/line-diff'

export interface IrDiffTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

// 行级 diff 实现在 ../ui/chart/line-diff（真 LCS + 前后缀剥离 + 规模保护）。
// 旧的"极简 LCS"只同步公共前缀，顶部改一行就把整个文件标成增删，已废弃。

export function IrDiffTile(props: IrDiffTileProps): JSX.Element {
  const { tile, context, density } = props
  const filter = toQueryFilter(context)

  // 所有 hook 必须在任何 return 之前调用。
  // 之前这里先 `if (!context.pass) return`、再调 useQuery，属于在早返回后调 hook：
  // 从"未选 Pass"切到"已选 Pass"时 hook 数量变化，React 会直接抛错。
  const meta = useQuery(context.bundleId, { kind: 'meta' }, {})
  // 定位 pass 属于哪个阶段：靠名字猜会把 vectorize_loop 这类 TIR pass 归到 relay，
  // 路径拼错就永远找不到 artifact。这里用真实的 component 判断。
  const ranking = useQuery(context.bundleId, { kind: 'pass_ranking', limit: 500 }, {})

  // run_id 不该要求用户手动选：artifact 路径需要它，但 bundle 里就有。
  const runId = context.runId ?? meta.data?.runIds[0] ?? ''
  const passName = context.pass ?? ''
  const component = ranking.data?.items.find((i) => i.passName === passName)?.component ?? ''
  const stage: 'relay' | 'tir' | 'lower' =
    component === 'tir_pass' ? 'tir' : passName.includes('lower') ? 'lower' : 'relay'

  const ready = Boolean(passName && runId)
  const beforePath = ready ? irArtifactPath(runId, stage, passName, 'before') : ''
  const afterPath = ready ? irArtifactPath(runId, stage, passName, 'after') : ''

  const beforeQuery = useQuery(
    context.bundleId,
    ready ? { kind: 'artifact', path: beforePath } : null,
    filter,
    ready,
  )
  const afterQuery = useQuery(
    context.bundleId,
    ready ? { kind: 'artifact', path: afterPath } : null,
    filter,
    ready,
  )

  const diff = useMemo(() => {
    if (!beforeQuery.data || !afterQuery.data) return null
    const before = beforeQuery.data.content?.split('\n') ?? []
    const after = afterQuery.data.content?.split('\n') ?? []
    return lcsDiff(before, after)
  }, [beforeQuery.data, afterQuery.data])

  if (!ready) {
    return (
      <div className="tile-content">
        请先选择一个 Pass 查看 IR diff
        {!passName && ranking.data && ranking.data.items.length > 0 && (
          <p className="note">
            可选：{ranking.data.items.slice(0, 6).map((i) => i.passName).join('、')}
          </p>
        )}
      </div>
    )
  }

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
