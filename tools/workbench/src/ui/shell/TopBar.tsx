import { useWorkbench, useGlobalContext } from '../../state/store'
import './shell.css'

/**
 * 顶栏组件（需求 §5.1、§19.3）。
 * 固定显示：Workspace 名、Bundle、Baseline、Candidate、Model、Input Shape、
 * Target、Device、Run、Time Range、搜索、Strip/Gather/Overview 切换、保存状态、命令面板。
 * 必须显示当前数据位置（本地或服务）并强调隐私。
 */
export function TopBar(): JSX.Element {
  const ws = useWorkbench((s) => s.activeWorkspace())
  // 必须用 useGlobalContext，不能写成 selector 里调 globalContext()——那样每次返回
  // 新对象，会把 zustand 的快照比较打穿，导致无限重渲染。
  const globalContext = useGlobalContext()
  const bundles = useWorkbench((s) => s.bundles)
  const dirty = useWorkbench((s) => s.dirty)
  const mode = useWorkbench((s) => s.mode)
  const setMode = useWorkbench((s) => s.setMode)
  const setCommandPaletteOpen = useWorkbench((s) => s.setCommandPaletteOpen)

  if (!ws) return <></>

  // 获取当前 bundle 和 baseline bundle 的元数据
  const bundleId = globalContext.bundleId
  const baselineBundleId = globalContext.baselineBundleId
  const bundleRef = bundleId ? bundles[bundleId]?.ref : null
  const baselineRef = baselineBundleId ? bundles[baselineBundleId]?.ref : null

  // 构造显示内容：Bundle 名称或路径摘要
  const getBundleDisplay = () => {
    if (!bundleId) return '—'
    if (bundleRef) {
      const label = bundleRef.label || bundleRef.id
      return label.length > 20 ? `${label.substring(0, 17)}…` : label
    }
    return bundleId.substring(0, 8)
  }

  const getBaselineDisplay = () => {
    if (!baselineBundleId) return '—'
    if (baselineRef) {
      const label = baselineRef.label || baselineRef.id
      return label.length > 20 ? `${label.substring(0, 17)}…` : label
    }
    return baselineBundleId.substring(0, 8)
  }

  // 时间范围显示（秒级摘要）
  const getTimeRangeDisplay = () => {
    const tr = globalContext.timeRange
    if (!tr) return '全部'
    const start = (tr.startNs / 1e9).toFixed(2)
    const end = (tr.endNs / 1e9).toFixed(2)
    return `${start}s - ${end}s`
  }

  return (
    <div className="topbar">
      <div className="topbar-start">
        {/* Workspace 名称 */}
        <div className="topbar-context-item">
          <span className="topbar-context-label">Workspace:</span>
          <span className="topbar-context-value">{ws.name}</span>
        </div>

        {/* Bundle 显示和数据位置提示（§19.3） */}
        <div className="topbar-context-item" title={`数据来源：${bundleRef?.hint || '本地'}`}>
          <span className="topbar-context-label">Bundle:</span>
          <span className="topbar-context-value">{getBundleDisplay()}</span>
          {bundleRef && <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>🔒 本地</span>}
        </div>

        {/* Baseline（Compare 时显示） */}
        {ws.compareEnabled && (
          <div className="topbar-context-item">
            <span className="topbar-context-label">Baseline:</span>
            <span className="topbar-context-value">{getBaselineDisplay()}</span>
          </div>
        )}
      </div>

      <div className="topbar-center">
        {/* 时间范围显示与清除 */}
        <div className="topbar-context-item">
          <span className="topbar-context-label">Range:</span>
          <span className="topbar-context-value">{getTimeRangeDisplay()}</span>
        </div>

        {/* 搜索框 */}
        <input
          type="text"
          placeholder="搜索（§15 未实现）"
          disabled
          style={{ width: '120px' }}
          aria-label="全局搜索"
        />
      </div>

      <div className="topbar-end">
        {/* Strip/Gather/Overview 切换 */}
        <div style={{ display: 'flex', gap: '2px', borderRight: '1px solid var(--border)', paddingRight: '8px' }}>
          <button
            className={mode === 'strip' ? 'active' : ''}
            onClick={() => setMode('strip')}
            title="Strip 模式 (H)"
            aria-label="进入 Strip 模式"
            style={{
              background: mode === 'strip' ? 'var(--bg-raised)' : 'transparent',
              fontSize: '11px',
            }}
          >
            Strip
          </button>
          <button
            className={mode === 'gather' ? 'active' : ''}
            onClick={() => setMode('gather')}
            title="Gather 模式 (G)"
            aria-label="进入 Gather 模式"
            style={{
              background: mode === 'gather' ? 'var(--bg-raised)' : 'transparent',
              fontSize: '11px',
            }}
          >
            Gather
          </button>
          <button
            className={mode === 'overview' ? 'active' : ''}
            onClick={() => setMode('overview')}
            title="Overview 模式 (O)"
            aria-label="进入 Overview 模式"
            style={{
              background: mode === 'overview' ? 'var(--bg-raised)' : 'transparent',
              fontSize: '11px',
            }}
          >
            Overview
          </button>
        </div>

        {/* 保存状态 */}
        {dirty && (
          <div style={{ fontSize: '11px', color: 'var(--sem-warn)', fontWeight: 500 }}>
            未保存
          </div>
        )}

        {/* 命令面板入口 */}
        <button
          onClick={() => setCommandPaletteOpen(true)}
          title="打开命令面板 (Ctrl+K)"
          aria-label="打开命令面板"
          style={{ fontSize: '11px' }}
        >
          ⌘K
        </button>
      </div>
    </div>
  )
}
