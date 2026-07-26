import { useState } from 'react'
import { useWorkbench } from '../../state/store'
import './shell.css'

/**
 * WorkspaceRail 组件（需求 §5.2、§12.4）。
 * 显示 Workspace 列表、拖动排序、每项显示告警数、未保存标记、当前活动高亮。
 * 必须可折叠（折叠后只剩窄条图标）。
 */
export function WorkspaceRail(): JSX.Element {
  const doc = useWorkbench((s) => s.doc)
  const activeWorkspaceId = useWorkbench((s) => s.doc.activeWorkspaceId)
  const switchWorkspace = useWorkbench((s) => s.switchWorkspace)
  const createWorkspace = useWorkbench((s) => s.createWorkspace)
  const reorderWorkspace = useWorkbench((s) => s.reorderWorkspace)

  const [collapsed, setCollapsed] = useState(false)
  const [dragFrom, setDragFrom] = useState<string | null>(null)

  const workspaces = doc.workspaceOrder
    .map((id) => doc.workspaces[id])
    .filter((w): w is typeof doc.workspaces[string] => Boolean(w))

  const handleDragStart = (id: string) => {
    setDragFrom(id)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (id: string) => {
    if (!dragFrom || dragFrom === id) {
      setDragFrom(null)
      return
    }
    const fromIdx = doc.workspaceOrder.indexOf(dragFrom)
    const toIdx = doc.workspaceOrder.indexOf(id)
    if (fromIdx >= 0 && toIdx >= 0) {
      // 拖到的目标位置
      const targetIdx = fromIdx < toIdx ? toIdx : toIdx
      reorderWorkspace(dragFrom, targetIdx)
    }
    setDragFrom(null)
  }

  return (
    <div className={`workspace-rail ${collapsed ? 'collapsed' : ''}`}>
      {/* 栏头：标题 + 折叠按钮 */}
      <div className="workspace-rail-header">
        <div className="workspace-rail-title">Workspaces</div>
        <button
          className="workspace-rail-toggle"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? '展开' : '折叠'}
          aria-label={collapsed ? '展开工作区栏' : '折叠工作区栏'}
        >
          {collapsed ? '▶' : '◀'}
        </button>
      </div>

      {/* Workspace 列表 */}
      <div className="workspace-rail-list">
        {workspaces.map((ws) => {
          const isDirty = ws.updatedAt > (ws.createdAt || 0) // 简化检查：已修改过
          // TODO: 从诊断数据统计告警数
          const warningCount = 0

          return (
            <div
              key={ws.id}
              className={`workspace-item ${ws.id === activeWorkspaceId ? 'active' : ''}`}
              onClick={() => switchWorkspace(ws.id)}
              draggable
              onDragStart={() => handleDragStart(ws.id)}
              onDragOver={handleDragOver}
              onDrop={() => handleDrop(ws.id)}
              role="button"
              tabIndex={0}
              aria-selected={ws.id === activeWorkspaceId}
              aria-label={`${ws.name}${isDirty ? ' (未保存)' : ''}${warningCount > 0 ? ` (${warningCount} 个告警)` : ''}`}
            >
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {ws.name}
              </span>
              {isDirty && <div className="workspace-item-dirty" title="未保存" />}
              {warningCount > 0 && <span className="workspace-item-badge">{warningCount}</span>}
            </div>
          )
        })}
      </div>

      {/* 快速操作按钮 */}
      {!collapsed && (
        <div style={{ padding: '8px 4px', borderTop: '1px solid var(--border)', display: 'flex', gap: '4px' }}>
          <button
            onClick={() => createWorkspace()}
            title="新建 Workspace"
            aria-label="新建 Workspace"
            style={{ flex: 1, fontSize: '11px' }}
          >
            + 新建
          </button>
        </div>
      )}
    </div>
  )
}
