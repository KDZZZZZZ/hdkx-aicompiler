/**
 * Tile 层的 store 访问助手。
 *
 * 封装 useWorkbench 获取，并统一数据流。
 */

import type { WorkbenchState } from '../state/store'
import { useWorkbench } from '../state/store'

export function useStore<T>(selector: (s: WorkbenchState) => T): T {
  return useWorkbench(selector)
}
