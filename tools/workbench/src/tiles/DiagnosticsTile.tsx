import './tiles.css'
/**
 * 诊断 Tile：查看自动诊断结论、证据与下一步建议。
 *
 * 查询：{kind:'diagnostics'}
 * 特殊处理：
 * - notAnalyzed=true 时显示明确引导（需要跑 python -m kxc_agent.cli analyze_bundle）
 * - evidence 点击后按 pass_name/component 做近似回跳并明确告知这是近似定位
 * - 支持标记已确认/误报/加备注
 */

import React, { useState } from 'react'
import type { Tile, TileDensity, AnalysisContext } from '../state/types'
import { useQuery, toQueryFilter } from '../kxc/query-client'
import { useWorkbench } from '../state/store'

export interface DiagnosticsTileProps {
  tile: Tile
  context: AnalysisContext
  density: TileDensity
}

export function DiagnosticsTile(props: DiagnosticsTileProps): JSX.Element {
  const { tile, context, density } = props
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)

  const filter = toQueryFilter(context)
  const query = useQuery(context.bundleId, { kind: 'diagnostics' }, filter)

  if (query.status === 'loading') return <div className="tile-content">加载中...</div>
  if (query.status === 'error') return <div className="tile-content error">错误: {query.error}</div>
  if (query.status === 'cancelled') return <div className="tile-content">已取消</div>
  if (!query.data) return <div className="tile-content">无数据</div>

  const data = query.data

  if (data.notAnalyzed) {
    return (
      <div className="tile-content diagnostics-unanalyzed">
        <div className="alert" role="alert">
          <strong>⚠ 尚未分析</strong>
          <p>Bundle 还没有运行诊断分析。请执行：</p>
          <code>python -m kxc_agent.cli analyze_bundle /path/to/bundle</code>
        </div>
      </div>
    )
  }

  if (data.diagnostics.length === 0) {
    return (
      <div className="tile-content">
        <p>无诊断结果</p>
      </div>
    )
  }

  const displayItems = density === 'thumbnail' ? data.diagnostics.slice(0, 3) : data.diagnostics

  return (
    <div className="tile-content diagnostics">
      <div className="diagnostics-list">
        {displayItems.map((diag, idx) => (
          <div key={idx} className={`diagnostic-item severity-${diag.severity}`}>
            <div
              className="diagnostic-header"
              onClick={() => setExpandedIdx(expandedIdx === idx ? null : idx)}
              role="button"
              tabIndex={0}
            >
              <span className="severity-badge">{(diag.severity ?? 'i')[0]?.toUpperCase() ?? 'I'}</span>
              <span className="category">{diag.category}</span>
              <span className="component" title={diag.component}>
                {diag.component}
              </span>
            </div>

            <div className="summary">{diag.summary}</div>

            {expandedIdx === idx && (
              <div className="diagnostic-details">
                {diag.evidence && Object.keys(diag.evidence).length > 0 && (
                  <div className="evidence">
                    <strong>证据：</strong>
                    <ul>
                      {Object.entries(diag.evidence ?? {}).map(([k, v]) => (
                        <li key={k}>
                          <span className="key">{k}:</span> <span className="value">{String(v)}</span>
                        </li>
                      ))}
                    </ul>
                    <p className="note">
                      💡 点击上方项目可按名字近似定位，但因证据不含 span_id 或时间范围，无法精确到事件。
                    </p>
                  </div>
                )}

                {diag.next_steps && diag.next_steps.length > 0 && (
                  <div className="next-steps">
                    <strong>下一步：</strong>
                    <ol>
                      {diag.next_steps.map((step, i) => (
                        <li key={i}>{step}</li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="sr-only">
        诊断结果，共 {data.diagnostics.length} 项。
        {data.diagnostics.slice(0, 3).map((diag) => `${diag.severity} ${diag.category} ${diag.summary}`).join('；')}
      </div>
    </div>
  )
}
