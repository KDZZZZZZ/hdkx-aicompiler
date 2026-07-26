import { useWorkbench, useGlobalContext } from '../../state/store'
import { useBundleActions } from '../bundle/useBundleActions'
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
  const bindBundle = useWorkbench((s) => s.bindBundle)
  const bindBaseline = useWorkbench((s) => s.bindBaseline)
  const setGlobalFilter = useWorkbench((s) => s.setGlobalFilter)
  const bundle = useBundleActions()
  const loadedIds = Object.keys(bundles)

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

        {/* Bundle 选择、加载与数据位置提示（§2.1、§19.3） */}
        <div className="topbar-context-item" title={`数据来源：${bundleRef?.hint || '尚未加载'}`}>
          <span className="topbar-context-label">Bundle:</span>
          <select
            className="topbar-select"
            aria-label="选择当前 Bundle"
            value={ws.bundleId ?? ''}
            disabled={bundle.loading}
            onChange={(e) => {
              const v = e.target.value
              if (!v) return
              // 已加载过的直接绑定，没加载过的先按样例 id 拉取。
              // 加载函数失败会 rethrow（agent 链路靠它判断成败）；UI 侧 toast 已发，吞掉即可。
              if (loadedIds.includes(v)) bindBundle(v)
              else void bundle.loadFixtureById(v).catch(() => {})
            }}
          >
            <option value="">{bundle.loading ? '加载中…' : '— 选择 Bundle —'}</option>
            {loadedIds.length > 0 && (
              <optgroup label="已加载">
                {loadedIds.map((id) => (
                  <option key={id} value={id}>
                    {bundles[id]?.ref.label ?? id}
                  </option>
                ))}
              </optgroup>
            )}
            {bundle.fixtures.length > 0 && (
              <optgroup label="样例 / 本地开发服务">
                {bundle.fixtures.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.id}（{f.event_count} 事件{f.variant === 'real' ? '，真实产物' : ''}）
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          <button
            onClick={() => void bundle.openDirectory().catch(() => {})}
            disabled={bundle.loading}
            title="从本机选择一个 KXC profiling bundle 目录；文件不会离开本机"
            aria-label="打开本地 bundle 目录"
            style={{ fontSize: '11px' }}
          >
            打开目录…
          </button>
          {bundleRef && (
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }} title={bundleRef.hint}>
              🔒 本地
            </span>
          )}
        </div>

        {bundle.error && (
          <div className="topbar-context-item" role="alert" style={{ color: 'var(--sem-error)' }}>
            ⚠ {bundle.error}
          </div>
        )}

        {/* Baseline：选了就进入 Compare（§4.5） */}
        <div className="topbar-context-item">
          <span className="topbar-context-label">Baseline:</span>
          <select
            className="topbar-select"
            aria-label="选择 Baseline Bundle"
            value={ws.baselineBundleId ?? ''}
            onChange={(e) => bindBaseline(e.target.value || null)}
          >
            <option value="">— 无 —</option>
            {loadedIds
              .filter((id) => id !== ws.bundleId)
              .map((id) => (
                <option key={id} value={id}>
                  {bundles[id]?.ref.label ?? id}
                </option>
              ))}
          </select>
          {ws.compareEnabled && (
            <span style={{ fontSize: '11px', color: 'var(--cmp-candidate)' }}>
              对比中：{getBaselineDisplay()}
            </span>
          )}
        </div>
      </div>

      <div className="topbar-center">
        {/* 时间范围显示与清除 */}
        <div className="topbar-context-item">
          <span className="topbar-context-label">Range:</span>
          <span className="topbar-context-value">{getTimeRangeDisplay()}</span>
          {globalContext.timeRange && (
            <button
              onClick={() => setGlobalFilter({ timeRange: null })}
              title="清除时间范围过滤"
              aria-label="清除时间范围过滤"
              style={{ fontSize: '11px' }}
            >
              ✕
            </button>
          )}
        </div>

        {/* 全局搜索：写入 context.search，所有 Tile 的查询都会带上它 */}
        <input
          type="search"
          placeholder="搜索 pass / 算子 / kernel / 消息"
          value={globalContext.search ?? ''}
          onChange={(e) => setGlobalFilter({ search: e.target.value || null })}
          style={{ width: '200px' }}
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
