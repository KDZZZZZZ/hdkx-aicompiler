/**
 * 键位配置表（需求 §14）。
 *
 * 默认映射 + localStorage 覆盖 + 可配置。
 * 所有快捷键必须可关闭（开关存在 localStorage）。
 */

export interface KeymapEntry {
  id: string
  name: string
  description: string
  keys: string[]
  /** 是否可重新映射 */
  customizable: boolean
  /** 输入框内不得触发 */
  blockInInput: boolean
}

/** 键位 ID，必须与命令面板 / action 一一对应 */
export type KeymapId =
  | 'nav-left'
  | 'nav-right'
  | 'nav-up'
  | 'nav-down'
  | 'drill-down'
  | 'drill-as-tab'
  | 'consume-left'
  | 'consume-right'
  | 'expel'
  | 'pin'
  | 'focus'
  | 'gather-toggle'
  | 'overview-toggle'
  | 'compare'
  | 'search'
  | 'command-palette'
  | 'escape'
  | 'undo'
  | 'redo'

export const DEFAULT_KEYMAP: Record<KeymapId, KeymapEntry> = {
  'nav-left': {
    id: 'nav-left',
    name: '上一列',
    description: '导航到左侧列',
    keys: ['h'],
    customizable: true,
    blockInInput: true,
  },
  'nav-right': {
    id: 'nav-right',
    name: '下一列',
    description: '导航到右侧列',
    keys: ['l'],
    customizable: true,
    blockInInput: true,
  },
  'nav-up': {
    id: 'nav-up',
    name: '上一个 Tile',
    description: '同列向上移动聚焦',
    keys: ['j'],
    customizable: true,
    blockInInput: true,
  },
  'nav-down': {
    id: 'nav-down',
    name: '下一个 Tile',
    description: '同列向下移动聚焦',
    keys: ['k'],
    customizable: true,
    blockInInput: true,
  },
  'drill-down': {
    id: 'drill-down',
    name: 'Drill Down',
    description: '深入到选中对象，在右侧新建列',
    keys: ['Enter'],
    customizable: true,
    blockInInput: true,
  },
  'drill-as-tab': {
    id: 'drill-as-tab',
    name: '作为 Tab 打开',
    description: 'Drill Down 到当前列的 Tab 而非新列',
    keys: ['Shift+Enter'],
    customizable: true,
    blockInInput: true,
  },
  'consume-left': {
    id: 'consume-left',
    name: 'Consume 到左列',
    description: '将 Tile 收入左侧列',
    keys: ['['],
    customizable: true,
    blockInInput: true,
  },
  'consume-right': {
    id: 'consume-right',
    name: 'Consume 到右列',
    description: '将 Tile 收入右侧列',
    keys: [']'],
    customizable: true,
    blockInInput: true,
  },
  expel: {
    id: 'expel',
    name: 'Expel',
    description: '将 Tile 从列中拆出为新列',
    keys: ['e'],
    customizable: true,
    blockInInput: true,
  },
  pin: {
    id: 'pin',
    name: 'Pin',
    description: '固定当前列',
    keys: ['p'],
    customizable: true,
    blockInInput: true,
  },
  focus: {
    id: 'focus',
    name: 'Focus',
    description: '全屏查看当前 Tile / Column',
    keys: ['f'],
    customizable: true,
    blockInInput: true,
  },
  'gather-toggle': {
    id: 'gather-toggle',
    name: '进出 Gather',
    description: '切换 Gather 模式',
    keys: ['g'],
    customizable: true,
    blockInInput: true,
  },
  'overview-toggle': {
    id: 'overview-toggle',
    name: '进出 Overview',
    description: '切换 Overview 模式',
    keys: ['o'],
    customizable: true,
    blockInInput: true,
  },
  compare: {
    id: 'compare',
    name: 'Compare',
    description: '开启对比模式',
    keys: ['c'],
    customizable: true,
    blockInInput: true,
  },
  search: {
    id: 'search',
    name: '搜索',
    description: '打开搜索框',
    keys: ['/'],
    customizable: true,
    blockInInput: true,
  },
  'command-palette': {
    id: 'command-palette',
    name: '命令面板',
    description: '打开命令面板',
    keys: ['Control+k', 'Meta+k'],
    customizable: false, // 不允许改 Cmd+K，太常见
    blockInInput: false, // 命令面板允许在输入框内打开
  },
  escape: {
    id: 'escape',
    name: '返回上一级',
    description: '关闭面板、退出模式、清选择',
    keys: ['Escape'],
    customizable: false, // Esc 不允许改
    blockInInput: false,
  },
  undo: {
    id: 'undo',
    name: '撤销',
    description: '撤销上一操作',
    keys: ['Control+z', 'Meta+z'],
    customizable: false,
    blockInInput: false,
  },
  redo: {
    id: 'redo',
    name: '重做',
    description: '重做被撤销的操作',
    keys: ['Control+Shift+z', 'Meta+Shift+z'],
    customizable: false,
    blockInInput: false,
  },
}

const KEYMAP_STORAGE_KEY = 'kxc-keymap-v1'
const KEYMAP_DISABLED_KEY = 'kxc-keymap-disabled'

/**
 * 从 localStorage 读取用户自定义的按键映射。
 * 返回完整的 keymap，缺失项用默认值填充。
 */
export function loadKeymap(): Record<KeymapId, KeymapEntry> {
  try {
    const stored = localStorage.getItem(KEYMAP_STORAGE_KEY)
    if (!stored) return { ...DEFAULT_KEYMAP }
    const custom = JSON.parse(stored) as Partial<Record<KeymapId, KeymapEntry>>
    return { ...DEFAULT_KEYMAP, ...custom }
  } catch {
    return { ...DEFAULT_KEYMAP }
  }
}

/**
 * 保存用户自定义的按键映射。
 * 只存储被修改的项，减少存储。
 */
export function saveKeymap(custom: Partial<Record<KeymapId, KeymapEntry>>): void {
  try {
    localStorage.setItem(KEYMAP_STORAGE_KEY, JSON.stringify(custom))
  } catch {
    console.error('Failed to save keymap')
  }
}

/**
 * 检查快捷键是否被禁用。
 */
export function isKeymapDisabled(): boolean {
  try {
    return localStorage.getItem(KEYMAP_DISABLED_KEY) === 'true'
  } catch {
    return false
  }
}

/**
 * 切换快捷键启用/禁用状态。
 */
export function setKeymapDisabled(disabled: boolean): void {
  try {
    localStorage.setItem(KEYMAP_DISABLED_KEY, disabled ? 'true' : 'false')
  } catch {
    console.error('Failed to save keymap disabled state')
  }
}

/**
 * 将按键字符串数组标准化为统一格式。
 * 例如：'control+k' → 'Control+k'
 */
export function normalizeKeys(keys: string[]): string[] {
  return keys.map((k) => {
    const parts = k.split('+')
    return parts
      .map((p, i) => {
        const lower = p.toLowerCase()
        // 修饰符首字母大写，其他小写
        if (
          lower === 'control' ||
          lower === 'shift' ||
          lower === 'alt' ||
          lower === 'meta'
        ) {
          return lower.charAt(0).toUpperCase() + lower.slice(1)
        }
        // 特殊键首字母大写
        if (
          lower === 'enter' ||
          lower === 'escape' ||
          lower === 'backspace' ||
          lower === 'tab' ||
          lower === 'delete'
        ) {
          return lower.charAt(0).toUpperCase() + lower.slice(1)
        }
        // 普通单字母小写
        return lower
      })
      .join('+')
  })
}

/**
 * 从键盘事件中提取规范化的按键组合。
 */
export function extractKeyCombo(ev: KeyboardEvent): string {
  const parts: string[] = []
  if (ev.ctrlKey) parts.push('Control')
  if (ev.shiftKey) parts.push('Shift')
  if (ev.altKey) parts.push('Alt')
  if (ev.metaKey) parts.push('Meta')

  const key = ev.key === ' ' ? 'Space' : ev.key
  if (key && !['Control', 'Shift', 'Alt', 'Meta'].includes(key)) {
    // 单字母小写，其他特殊键首字母大写
    if (key.length === 1) {
      parts.push(key.toLowerCase())
    } else {
      parts.push(key.charAt(0).toUpperCase() + key.slice(1))
    }
  }
  return parts.join('+')
}

/**
 * 检查事件是否匹配某个键位配置。
 */
export function matchesKeyCombo(ev: KeyboardEvent, keys: string[]): boolean {
  const combo = extractKeyCombo(ev)
  // 标准化后比较
  const normalized = normalizeKeys(keys)
  return normalized.some((k) => k === combo)
}
