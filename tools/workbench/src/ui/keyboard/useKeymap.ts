/**
 * 键盘快捷键（需求 §14）。
 *
 * 在 App 级别注册全局键盘监听，完整的快捷键表在 keymap-config.ts。
 *
 * 关键细节：
 * 1. 输入框/textarea/contenteditable 内不得触发单键快捷键（H/L/J/K/E/P/F/G/O/C）
 * 2. Cmd/Ctrl+K 与 Cmd/Ctrl+Z 等系统快捷键始终生效
 * 3. 快捷键可通过 keymap-config.isKeymapDisabled() 全局关闭
 * 4. Esc 是"返回上一级"：命令面板开着→关；Focus→退出；Gather/Overview→回 Strip；有选择→清选择
 */

import { useEffect } from 'react'
import { useWorkbench } from '../../state/store'
import {
  DEFAULT_KEYMAP,
  isKeymapDisabled,
  matchesKeyCombo,
  type KeymapId,
} from './keymap-config'

export function useKeymap(): void {
  const store = useWorkbench()

  // 派生状态便于检查输入框 focus
  function isEditingInput(target: EventTarget | null): boolean {
    if (!target) return false
    const el = target as HTMLElement
    const tag = el.tagName?.toUpperCase()
    return (
      tag === 'INPUT' ||
      tag === 'TEXTAREA' ||
      el.getAttribute('contenteditable') === 'true'
    )
  }

  // 检查是否应该处理这个快捷键
  function shouldHandle(ev: KeyboardEvent, keymapId: KeymapId): boolean {
    // 快捷键全局禁用
    if (isKeymapDisabled()) return false

    const keyEntry = DEFAULT_KEYMAP[keymapId]
    if (!keyEntry) return false

    // 如果是"输入框内不得触发"的快捷键，且聚焦在编辑元素上，就不处理
    if (keyEntry.blockInInput && isEditingInput(ev.target)) {
      return false
    }

    return matchesKeyCombo(ev, keyEntry.keys)
  }

  // 命令面板：全局 Ctrl/Cmd+K，即使在输入框内也行
  function handleCommandPalette(ev: KeyboardEvent): void {
    if (matchesKeyCombo(ev, DEFAULT_KEYMAP['command-palette'].keys)) {
      ev.preventDefault()
      store.setCommandPaletteOpen(!store.commandPaletteOpen)
    }
  }

  // Esc：层级返回（§14 Esc）
  function handleEscape(ev: KeyboardEvent): void {
    if (!matchesKeyCombo(ev, DEFAULT_KEYMAP.escape.keys)) return
    if (isKeymapDisabled()) return

    ev.preventDefault()

    // 优先级：命令面板 > Focus > Gather/Overview > 清选择
    if (store.commandPaletteOpen) {
      store.setCommandPaletteOpen(false)
      return
    }

    const mode = store.mode
    if (mode === 'focus') {
      store.exitFocus()
      return
    }

    if (mode === ('gather' as const) || mode === ('overview' as const)) {
      store.setMode('strip')
      return
    }

    // 清选择
    store.setHover(null)
  }

  // Undo/Redo
  function handleUndoRedo(ev: KeyboardEvent): void {
    if (matchesKeyCombo(ev, DEFAULT_KEYMAP.undo.keys)) {
      ev.preventDefault()
      store.undo()
    } else if (matchesKeyCombo(ev, DEFAULT_KEYMAP.redo.keys)) {
      ev.preventDefault()
      store.redo()
    }
  }

  // 导航与编辑快捷键（§14）
  // 这些键只在 Strip 模式下有效，且不在输入框内
  function handleStripNavigation(ev: KeyboardEvent): void {
    if (store.mode !== 'strip') return

    const focused = store.activeWorkspace()
    if (!focused) return

    const cols = store.columnsOf(focused.id)
    if (cols.length === 0) return

    // 左右导航列（H/L）
    if (shouldHandle(ev, 'nav-left')) {
      ev.preventDefault()
      const currentIdx = cols.findIndex((c) => c.id === focused.focusedColumnId)
      if (currentIdx > 0) {
        const prevCol = cols[currentIdx - 1]!
        store.focusColumn(prevCol.id)
      }
      return
    }

    if (shouldHandle(ev, 'nav-right')) {
      ev.preventDefault()
      const currentIdx = cols.findIndex((c) => c.id === focused.focusedColumnId)
      if (currentIdx < cols.length - 1) {
        const nextCol = cols[currentIdx + 1]!
        store.focusColumn(nextCol.id)
      }
      return
    }

    // 当前列内上下导航 Tile
    const focusedColId = focused.focusedColumnId
    if (!focusedColId) return

    const tiles = store.tilesOf(focusedColId)
    if (tiles.length === 0) return

    const col = focused.columnIds
      .map((id) => store.doc.columns[id]!)
      .find((c) => c.id === focusedColId)
    if (!col) return

    const activeTileId = col.activeTileId ?? tiles[0]?.id
    const currentIdx = tiles.findIndex((t) => t.id === activeTileId)

    if (shouldHandle(ev, 'nav-up')) {
      ev.preventDefault()
      if (currentIdx > 0) {
        store.setActiveTile(focusedColId, tiles[currentIdx - 1]!.id)
      }
      return
    }

    if (shouldHandle(ev, 'nav-down')) {
      ev.preventDefault()
      if (currentIdx < tiles.length - 1) {
        store.setActiveTile(focusedColId, tiles[currentIdx + 1]!.id)
      }
      return
    }
  }

  // 当前聚焦 Tile 的操作
  function handleTileActions(ev: KeyboardEvent): void {
    if (store.mode !== 'strip') return

    const focused = store.activeWorkspace()
    if (!focused) return

    const focusedColId = focused.focusedColumnId
    if (!focusedColId) return

    const col = store.doc.columns[focusedColId]
    if (!col) return

    const activeTileId = col.activeTileId
    if (!activeTileId) return

    const tile = store.doc.tiles[activeTileId]
    if (!tile) return

    // Drill Down
    if (shouldHandle(ev, 'drill-down')) {
      ev.preventDefault()
      if (tile.selection) {
        store.drillDown(activeTileId, tile.selection, false)
      }
      return
    }

    // Drill as Tab
    if (shouldHandle(ev, 'drill-as-tab')) {
      ev.preventDefault()
      if (tile.selection) {
        store.drillDown(activeTileId, tile.selection, true)
      }
      return
    }

    // Consume
    if (shouldHandle(ev, 'consume-left')) {
      ev.preventDefault()
      store.consume(activeTileId, 'left')
      return
    }

    if (shouldHandle(ev, 'consume-right')) {
      ev.preventDefault()
      store.consume(activeTileId, 'right')
      return
    }

    // Expel
    if (shouldHandle(ev, 'expel')) {
      ev.preventDefault()
      store.expel(activeTileId)
      return
    }

    // Pin Column
    if (shouldHandle(ev, 'pin')) {
      ev.preventDefault()
      store.togglePin(focusedColId)
      return
    }

    // Focus
    if (shouldHandle(ev, 'focus')) {
      ev.preventDefault()
      store.enterFocus('tile', activeTileId)
      return
    }

    // Gather（G）
    if (shouldHandle(ev, 'gather-toggle')) {
      ev.preventDefault()
      // 虽然当前是 strip 模式（由 handleStripNavigation 保证），但允许直接切到 gather
      store.setMode('gather')
      return
    }

    // Overview（O）
    if (shouldHandle(ev, 'overview-toggle')) {
      ev.preventDefault()
      // 虽然当前是 strip 模式（由 handleStripNavigation 保证），但允许直接切到 overview
      store.setMode('overview')
      return
    }

    // Compare
    if (shouldHandle(ev, 'compare')) {
      ev.preventDefault()
      store.setCompareEnabled(!focused.compareEnabled)
      return
    }

    // Search (打开搜索框，暂时只是 toast 提示，真实实现由 TopBar 或其他组件提供)
    if (shouldHandle(ev, 'search')) {
      ev.preventDefault()
      // TODO: 打开搜索框组件
      return
    }
  }

  useEffect(() => {
    const handleKeyDown = (ev: KeyboardEvent): void => {
      // 系统级快捷键始终处理
      handleCommandPalette(ev)
      handleEscape(ev)
      handleUndoRedo(ev)

      // 模式相关快捷键
      if (store.mode === 'strip') {
        handleStripNavigation(ev)
        handleTileActions(ev)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    store.mode,
    store.commandPaletteOpen,
    store.doc,
    store.activeWorkspace,
    store.columnsOf,
    store.tilesOf,
  ])
}
