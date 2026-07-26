import { useEffect } from 'react'
import { useWorkbench } from './state/store'
import { TopBar } from './ui/shell/TopBar'
import { WorkspaceRail } from './ui/shell/WorkspaceRail'
import { StripView } from './ui/shell/StripView'
import { Minimap } from './ui/shell/Minimap'
import { GatherView } from './ui/modes/GatherView'
import { OverviewView } from './ui/modes/OverviewView'
import { FocusView } from './ui/modes/FocusView'
import { CommandPalette } from './ui/command/CommandPalette'
import { useKeymap } from './ui/keyboard/useKeymap'
import { usePersistence } from './state/persist'
import { Toast } from './ui/shell/Toast'
import { useAgentBridge, AgentStatusBadge } from './ui/agent/useAgentBridge'
import './styles/tokens.css'
import './App.css'

export default function App() {
  const mode = useWorkbench((s) => s.mode)
  useKeymap()
  usePersistence()
  // 控制服务没起时它会安静退避重试，不影响正常使用。
  const agentStatus = useAgentBridge()

  // 顶层只切换主区域，TopBar / Rail / Minimap 在所有模式下保持可见，
  // 这样 Overview 与 Gather 也能一眼看到当前 bundle 与保存状态。
  return (
    <div className="app">
      <TopBar />
      <div className="app-body">
        <WorkspaceRail />
        <main className="app-main" aria-label="分析桌布主区域">
          {mode === 'strip' && <StripView />}
          {mode === 'gather' && <GatherView />}
          {mode === 'overview' && <OverviewView />}
          {mode === 'focus' && <FocusView />}
        </main>
      </div>
      {mode === 'strip' && <Minimap />}
      <CommandPalette />
      <Toast />
      <AgentStatusBadge status={agentStatus} />
    </div>
  )
}

/** 开发期辅助：把 store 挂到 window，便于在控制台核对状态。 */
export function useDevtoolsBridge() {
  useEffect(() => {
    if (import.meta.env.DEV) {
      ;(window as unknown as { __workbench: unknown }).__workbench = useWorkbench
    }
  }, [])
}
