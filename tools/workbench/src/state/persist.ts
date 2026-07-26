/**
 * 状态持久化（需求 §16、§19）。
 *
 * 三条路：
 * 1. 自动保存：doc 变化 debounce 500ms → localStorage
 * 2. 手动快照：saveLayoutSnapshot()、restoreLayoutSnapshot()、export()、import()
 * 3. 轻量 URL 状态：encodeUrlState() / applyUrlState()，只编码 workspace id + mode + 聚焦列 + 全局过滤器
 *
 * 关键细节：
 * - exportSnapshot() 本身就只导出 doc（src/state/store.ts:184），不会额外塞大数据
 * - Bundle 路径失效提示（§16 最后）：恢复后若 bundleId 不在已加载列表里，toast 提示
 * - URL 轻量状态分享前必须走 sanitizeForShare() 去掉敏感信息（§19.5）
 */

import { useEffect, useRef } from 'react'
import { useWorkbench } from './store'
import type { WorkbenchSnapshot, WorkbenchDoc, AnalysisContext } from './types'

const AUTOSAVE_STORAGE_KEY = 'kxc-workbench-doc-v1'
const SAVED_LAYOUTS_KEY = 'kxc-workbench-layouts-v1'
const DEBOUNCE_MS = 500

interface SavedLayout {
  id: string
  name: string
  doc: WorkbenchDoc
  savedAt: number
}

/**
 * 自动保存 + 启动恢复（§16、§19）。
 * 在 App 级别调一次。
 */
export function usePersistence(): void {
  // 只订阅 doc。绝不能写 `const store = useWorkbench()` 去订阅整个 store：
  // 那样任何一次 set() 都会换掉 store 引用，而下面的 effect 又会写 store，
  // 于是 effect → set → effect 无限循环（React 报 Maximum update depth exceeded）。
  // 需要调用 action 时一律走 getState()，它不参与订阅。
  const doc = useWorkbench((s) => s.doc)
  const debounceTimer = useRef<number | null>(null)
  const restored = useRef(false)

  // 启动恢复只做一次。空依赖 + ref 双保险：StrictMode 下 effect 会跑两遍，
  // 没有 ref 的话会把刚恢复的状态再导入一次，凭空多出一条 Undo 记录。
  useEffect(() => {
    if (restored.current) return
    restored.current = true
    try {
      const saved = localStorage.getItem(AUTOSAVE_STORAGE_KEY)
      if (!saved) return
      const snapshot = JSON.parse(saved) as WorkbenchSnapshot
      useWorkbench.getState().importSnapshot(snapshot)
    } catch (err) {
      console.error('恢复已保存布局失败:', err)
    }
  }, [])

  // 自动保存：doc 变化后 debounce 500ms
  useEffect(() => {
    if (debounceTimer.current !== null) clearTimeout(debounceTimer.current)

    debounceTimer.current = window.setTimeout(() => {
      try {
        const s = useWorkbench.getState()
        localStorage.setItem(AUTOSAVE_STORAGE_KEY, JSON.stringify(s.exportSnapshot()))
        // markSaved 只改 dirty，不动 doc，所以不会把本 effect 再次触发。
        s.markSaved()
      } catch (err) {
        console.error('自动保存失败:', err)
      }
    }, DEBOUNCE_MS)

    return () => {
      if (debounceTimer.current !== null) clearTimeout(debounceTimer.current)
    }
  }, [doc])
}

/**
 * 手动保存当前快照为命名 Layout。
 */
export function saveLayoutSnapshot(name: string): void {
  const store = useWorkbench.getState()
  try {
    const layouts = loadSavedLayouts()
    const id = `layout-${Date.now()}`
    layouts.push({
      id,
      name,
      doc: structuredClone(store.doc),
      savedAt: Date.now(),
    })
    localStorage.setItem(SAVED_LAYOUTS_KEY, JSON.stringify(layouts))
    store.showToast('info', `已保存布局 "${name}"`)
  } catch (err) {
    console.error('Failed to save layout:', err)
    store.showToast('error', '保存布局失败')
  }
}

/**
 * 恢复已保存的 Layout。
 */
export function restoreLayoutSnapshot(id: string): void {
  const store = useWorkbench.getState()
  try {
    const layouts = loadSavedLayouts()
    const layout = layouts.find((l) => l.id === id)
    if (!layout) {
      store.showToast('error', '布局不存在')
      return
    }

    // 检查 bundle 是否已加载
    const activeWsId = layout.doc.activeWorkspaceId
    const bundleId =
      activeWsId && layout.doc.workspaces[activeWsId]
        ? layout.doc.workspaces[activeWsId]?.bundleId
        : null
    if (bundleId && !store.bundles[bundleId]) {
      store.showToast('warn', `Bundle "${bundleId}" 未加载，请重新绑定`)
    }

    // 导入快照（构建完整 WorkbenchSnapshot）
    const snapshot: WorkbenchSnapshot = {
      kind: 'kxc-workbench-layout',
      version: 1,
      savedAt: layout.savedAt,
      doc: layout.doc,
      bundleRefs: bundleId ? [{ id: bundleId, label: bundleId, hint: '本地保存' }] : [],
    }
    store.importSnapshot(snapshot)
    store.showToast('info', `已恢复布局 "${layout.name}"`)
  } catch (err) {
    console.error('Failed to restore layout:', err)
    store.showToast('error', '恢复布局失败')
  }
}

/**
 * 导出当前视图为 JSON 文件。
 */
export function exportSnapshotToFile(): void {
  const store = useWorkbench.getState()
  try {
    const snapshot = store.exportSnapshot()
    const json = JSON.stringify(snapshot, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `workbench-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    store.showToast('info', '已导出快照')
  } catch (err) {
    console.error('Failed to export snapshot:', err)
    store.showToast('error', '导出快照失败')
  }
}

/**
 * 导入 JSON 快照文件。
 */
export async function importSnapshotFromFile(): Promise<void> {
  const store = useWorkbench.getState()
  try {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (ev) => {
      const file = (ev.target as HTMLInputElement).files?.[0]
      if (!file) return
      try {
        const text = await file.text()
        const snapshot = JSON.parse(text) as WorkbenchSnapshot
        store.importSnapshot(snapshot)
        store.showToast('info', '已导入快照')
      } catch (err) {
        console.error('Failed to parse imported file:', err)
        store.showToast('error', '导入快照失败：格式无效')
      }
    }
    input.click()
  } catch (err) {
    console.error('Failed to open file picker:', err)
    store.showToast('error', '打开文件选择器失败')
  }
}

/**
 * 编码轻量 URL 状态（§16、§19.5）。
 *
 * 只编码：
 * - 当前 workspace id
 * - 当前模式
 * - 聚焦列 id
 * - 全局过滤器（op、pass、kernel、shape 等关键过滤）
 *
 * 不编码：完整 doc（体积大、包含来源列 id）。
 */
export function encodeUrlState(): string {
  const store = useWorkbench.getState()
  const ws = store.activeWorkspace()
  if (!ws) return ''

  // 只取关键过滤键
  const filterable: Array<keyof AnalysisContext> = [
    'op',
    'pass',
    'kernel',
    'shapeSignature',
    'severity',
    'component',
  ]

  const filters: Record<string, string> = {}
  for (const key of filterable) {
    const val = ws.filters[key]
    if (val) filters[key] = String(val)
  }

  const state = {
    ws: ws.id,
    mode: store.mode,
    focusedCol: ws.focusedColumnId,
    filters,
  }

  // Base64 编码，便于 URL 友好
  const json = JSON.stringify(state)
  try {
    const encoded = btoa(json)
    return encoded
  } catch {
    return ''
  }
}

/**
 * 从 URL 应用轻量状态。
 * 返回是否成功应用。
 */
export function applyUrlState(): boolean {
  const store = useWorkbench.getState()
  try {
    const hash = window.location.hash.slice(1)
    if (!hash) return false

    const json = atob(hash)
    const state = JSON.parse(json) as {
      ws?: string
      mode?: string
      focusedCol?: string
      filters?: Record<string, string>
    }

    if (state.ws) {
      store.switchWorkspace(state.ws)
    }

    if (state.mode && ['strip', 'gather', 'overview', 'focus'].includes(state.mode)) {
      store.setMode(state.mode as any)
    }

    if (state.focusedCol) {
      store.focusColumn(state.focusedCol)
    }

    if (state.filters) {
      store.setGlobalFilter(state.filters)
    }

    return true
  } catch {
    return false
  }
}

/**
 * 加载已保存的 Layout 列表。
 */
function loadSavedLayouts(): SavedLayout[] {
  try {
    const stored = localStorage.getItem(SAVED_LAYOUTS_KEY)
    if (!stored) return []
    return JSON.parse(stored) as SavedLayout[]
  } catch {
    return []
  }
}

/**
 * 获取已保存的 Layout 列表（用于 UI 展示）。
 */
export function getSavedLayouts(): Array<{ id: string; name: string; savedAt: number }> {
  return loadSavedLayouts().map((l) => ({
    id: l.id,
    name: l.name,
    savedAt: l.savedAt,
  }))
}

/**
 * 删除已保存的 Layout。
 */
export function deleteSavedLayout(id: string): void {
  const store = useWorkbench.getState()
  try {
    let layouts = loadSavedLayouts()
    layouts = layouts.filter((l) => l.id !== id)
    localStorage.setItem(SAVED_LAYOUTS_KEY, JSON.stringify(layouts))
    store.showToast('info', '已删除布局')
  } catch (err) {
    console.error('Failed to delete layout:', err)
    store.showToast('error', '删除布局失败')
  }
}
