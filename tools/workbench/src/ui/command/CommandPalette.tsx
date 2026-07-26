/**
 * 命令面板（需求 §15）。
 *
 * Ctrl/Cmd+K 打开，模糊搜索并执行命令。
 * 全程支持键盘导航（上下、Enter 执行、Esc 关闭）。
 *
 * 关键细节：
 * - 命令定义集中在 commands.ts，这里只负责 UI 与交互
 * - 命令按分组显示
 * - 显示每条命令绑定的快捷键
 * - 执行后关闭面板
 */

import { useEffect, useState, useRef, useMemo } from 'react'
import { useWorkbench, useGlobalContext } from '../../state/store'
import { buildCommands, searchCommands, type Command, type CommandContext } from './commands'
import type { MetaResult } from '../../kxc/query-protocol'
import './command.css'

export function CommandPalette(): JSX.Element {
  // 只订阅真正会影响面板内容的几项。
  // 早先这里写的是 `const store = useWorkbench()`（订阅整个 store）
  // 外加在 useEffect 里 setState 派生命令列表，两者叠加会无限循环：
  // set() 换掉 store 引用 → effect 依赖变化 → setState → 重渲染 → 再次 set…
  // 派生数据用 useMemo，不要用 state + effect。
  const open = useWorkbench((s) => s.commandPaletteOpen)
  const mode = useWorkbench((s) => s.mode)
  const doc = useWorkbench((s) => s.doc)
  const bundles = useWorkbench((s) => s.bundles)
  const setCommandPaletteOpen = useWorkbench((s) => s.setCommandPaletteOpen)
  const globalContext = useGlobalContext()

  const [query, setQuery] = useState('')
  const [selectedIdx, setSelectedIdx] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const workspace = doc.activeWorkspaceId ? (doc.workspaces[doc.activeWorkspaceId] ?? null) : null
  const focusedColumn = workspace?.focusedColumnId
    ? (doc.columns[workspace.focusedColumnId] ?? null)
    : null
  const focusedTile = focusedColumn?.activeTileId
    ? (doc.tiles[focusedColumn.activeTileId] ?? null)
    : null
  const bundleMeta = workspace?.bundleId
    ? ((bundles[workspace.bundleId]?.meta as MetaResult | undefined) ?? null)
    : null

  const filteredCommands = useMemo(() => {
    const context: CommandContext = {
      // 命令执行时才需要 store，用 getState 拿最新的，避免把整个 store 拉进依赖。
      store: useWorkbench.getState(),
      workspace,
      focusedColumn,
      focusedTile,
      bundleMeta,
      globalContext,
      mode,
    }
    return searchCommands(buildCommands(context), query)
  }, [query, workspace, focusedColumn, focusedTile, bundleMeta, globalContext, mode])

  // 选中项越界时收回来（命令列表随搜索变短时会发生）
  const boundedIdx = Math.min(selectedIdx, Math.max(0, filteredCommands.length - 1))

  // 面板打开时清空搜索并聚焦输入框
  useEffect(() => {
    if (!open) return
    setQuery('')
    setSelectedIdx(0)
    const t = setTimeout(() => inputRef.current?.focus(), 0)
    return () => clearTimeout(t)
  }, [open])

  // 键盘导航
  useEffect(() => {
    if (!open) return

    const handleKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'ArrowDown') {
        ev.preventDefault()
        setSelectedIdx((i) => Math.min(i + 1, filteredCommands.length - 1))
      } else if (ev.key === 'ArrowUp') {
        ev.preventDefault()
        setSelectedIdx((i) => Math.max(i - 1, 0))
      } else if (ev.key === 'Enter') {
        ev.preventDefault()
        const cmd = filteredCommands[boundedIdx]
        if (cmd) {
          cmd.run()
          setCommandPaletteOpen(false)
        }
      } else if (ev.key === 'Escape') {
        ev.preventDefault()
        setCommandPaletteOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open, filteredCommands, boundedIdx, setCommandPaletteOpen])

  // 滚动到选中项
  useEffect(() => {
    if (!listRef.current) return
    const items = listRef.current.querySelectorAll('[data-command-id]')
    const selected = items[boundedIdx] as HTMLElement | undefined
    selected?.scrollIntoView({ block: 'nearest' })
  }, [boundedIdx])

  if (!open) return <></>

  // 按分组组织命令
  const grouped = new Map<string, Command[]>()
  for (const cmd of filteredCommands) {
    if (!grouped.has(cmd.group)) {
      grouped.set(cmd.group, [])
    }
    grouped.get(cmd.group)!.push(cmd)
  }

  let flatIdx = 0
  const groupedEntries = Array.from(grouped.entries())

  return (
    <div className="command-palette-overlay" onClick={() => setCommandPaletteOpen(false)}>
      <div className="command-palette" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          type="text"
          className="command-palette-input"
          placeholder="搜索命令... (按 Esc 关闭)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="command-palette-list" ref={listRef}>
          {filteredCommands.length === 0 ? (
            <div className="command-palette-empty">没有找到匹配的命令</div>
          ) : (
            groupedEntries.map(([group, cmds]) => (
              <div key={group} className="command-group">
                <div className="command-group-title">{group}</div>
                {cmds.map((cmd) => {
                  const isSelected = flatIdx === boundedIdx
                  flatIdx++
                  return (
                    <div
                      key={cmd.id}
                      data-command-id={cmd.id}
                      className={`command-item ${isSelected ? 'selected' : ''}`}
                      onClick={() => {
                        cmd.run()
                        setCommandPaletteOpen(false)
                      }}
                    >
                      <div className="command-item-main">
                        <div className="command-item-label">{cmd.label}</div>
                        {cmd.description && (
                          <div className="command-item-desc">{cmd.description}</div>
                        )}
                      </div>
                      {cmd.shortcut && (
                        <div className="command-item-shortcut">{cmd.shortcut}</div>
                      )}
                    </div>
                  )
                })}
              </div>
            ))
          )}
        </div>

        <div className="command-palette-footer">
          <div className="command-footer-hint">
            <kbd>↑</kbd>
            <kbd>↓</kbd>选择 • <kbd>Enter</kbd>执行 • <kbd>Esc</kbd>关闭
          </div>
        </div>
      </div>
    </div>
  )
}
