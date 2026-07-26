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

import { useEffect, useState, useRef } from 'react'
import { useWorkbench } from '../../state/store'
import { buildCommands, searchCommands, type Command, type CommandContext } from './commands'
import type { MetaResult } from '../../kxc/query-protocol'
import './command.css'

export function CommandPalette(): JSX.Element {
  const store = useWorkbench()
  const [query, setQuery] = useState('')
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [commands, setCommands] = useState<Command[]>([])
  const [filteredCommands, setFilteredCommands] = useState<Command[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // 获取当前上下文
  const workspace = store.activeWorkspace()
  const focusedColumn = workspace
    ? store.columnsOf(workspace.id).find((c) => c.id === workspace.focusedColumnId) ?? null
    : null
  const focusedTile = focusedColumn
    ? store.tilesOf(focusedColumn.id).find((t) => t.id === focusedColumn.activeTileId) ?? null
    : null

  const bundleMeta = workspace?.bundleId
    ? (store.bundles[workspace.bundleId]?.meta as MetaResult | undefined) ?? null
    : null

  const context: CommandContext = {
    store,
    workspace,
    focusedColumn,
    focusedTile,
    bundleMeta,
    globalContext: store.globalContext(),
    mode: store.mode,
  }

  // 构建和搜索命令
  useEffect(() => {
    const all = buildCommands(context)
    setCommands(all)

    const filtered = searchCommands(all, query)
    setFilteredCommands(filtered)
    setSelectedIdx(0)
  }, [query, workspace, focusedColumn, focusedTile, bundleMeta, store.mode])

  // 当面板打开时聚焦输入框
  useEffect(() => {
    if (store.commandPaletteOpen) {
      setQuery('')
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [store.commandPaletteOpen])

  // 键盘导航
  useEffect(() => {
    if (!store.commandPaletteOpen) return

    const handleKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'ArrowDown') {
        ev.preventDefault()
        setSelectedIdx((i) => Math.min(i + 1, filteredCommands.length - 1))
      } else if (ev.key === 'ArrowUp') {
        ev.preventDefault()
        setSelectedIdx((i) => Math.max(i - 1, 0))
      } else if (ev.key === 'Enter') {
        ev.preventDefault()
        const cmd = filteredCommands[selectedIdx]
        if (cmd) {
          cmd.run()
          store.setCommandPaletteOpen(false)
        }
      } else if (ev.key === 'Escape') {
        ev.preventDefault()
        store.setCommandPaletteOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [store.commandPaletteOpen, filteredCommands, selectedIdx, store])

  // 滚动到选中项
  useEffect(() => {
    if (!listRef.current) return
    const items = listRef.current.querySelectorAll('[data-command-id]')
    const selected = items[selectedIdx] as HTMLElement | undefined
    if (selected) {
      selected.scrollIntoView({ block: 'nearest' })
    }
  }, [selectedIdx])

  if (!store.commandPaletteOpen) return <></>

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
    <div className="command-palette-overlay" onClick={() => store.setCommandPaletteOpen(false)}>
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
                  const isSelected = flatIdx === selectedIdx
                  flatIdx++
                  return (
                    <div
                      key={cmd.id}
                      data-command-id={cmd.id}
                      className={`command-item ${isSelected ? 'selected' : ''}`}
                      onClick={() => {
                        cmd.run()
                        store.setCommandPaletteOpen(false)
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
