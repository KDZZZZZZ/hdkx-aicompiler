/**
 * Bundle 加载与切换（需求 §2.1、§19）。
 *
 * 提供：
 * - fixtures 自动列表
 * - 目录选择 (File System Access API)
 * - 加载流程与进度显示
 * - meta.malformedLines > 0 时的警告 toast
 * - Bundle 切换时的清理（调用 unload）
 *
 * 关键细节：
 * - 启动时自动 listFixtures() 拿到 fixtures 列表
 * - openDirectory() 走 pickBundleDirectory()，不支持时提示
 * - 加载完成后 registerBundle(ref, meta) + 若未绑定则自动 bindBundle
 * - malformedLines > 0 时必须 toast 警告
 * - 切换时调 unload() 清理旧 bundle（§18.3）
 */

import { useEffect, useState, useCallback } from 'react'
import { useWorkbench } from '../../state/store'
import {
  pickBundleDirectory,
  listFixtures,
  loadFromUrl,
  supportsDirectoryPicker,
  type BundleLoadResult,
} from '../../kxc/bundle-source'
import { getQueryClient } from '../../kxc/query-client'

export interface BundleActions {
  fixtures: Array<{
    id: string
    path: string
    variant: string
    event_count: number
  }>
  loading: boolean
  error: string | null
  openDirectory: () => Promise<void>
  loadFixtureById: (id: string) => Promise<void>
  setAsBaseline: (bundleId: string) => void
}

export function useBundleActions(): BundleActions {
  // 这里刻意不订阅 store：本 hook 只需要在回调里"写"，不需要跟着状态重渲染。
  // 订阅整个 store 会让每次 set() 都换掉引用，进而使依赖它的 useCallback/useEffect
  // 每轮都重建，很容易和 effect 里的 setState 组成死循环。
  const store = useWorkbench.getState()
  const [fixtures, setFixtures] = useState<
    Array<{ id: string; path: string; variant: string; event_count: number }>
  >([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const queryClient = getQueryClient()

  // 启动时加载 fixtures 列表
  useEffect(() => {
    const loadFixtures = async () => {
      try {
        const list = await listFixtures()
        setFixtures(list)
      } catch (err) {
        console.error('Failed to load fixtures:', err)
        // fixtures 加载失败不是致命错误，用户可以手选目录
      }
    }
    loadFixtures()
  }, [])

  // 统一的 bundle 加载流程
  const loadBundle = useCallback(
    async (result: BundleLoadResult) => {
      try {
        setLoading(true)
        setError(null)
        store.setLoading(result.ref.id)

        // 挂接加载进度显示
        const unsubscribe = queryClient.onProgress(({ parsed, total }) => {
          if (import.meta.env.DEV) {
            console.log(`Loading: ${parsed}/${total} events`)
          }
        })

        // 加载 bundle 数据
        const meta = await queryClient.load(result.ref.id, result.payload)

        unsubscribe()

        // 注册到 store
        store.registerBundle(result.ref, meta)

        // 若当前 workspace 未绑定 bundle，自动绑定
        const workspace = store.activeWorkspace()
        if (workspace && !workspace.bundleId) {
          store.bindBundle(result.ref.id)
        }

        // 检查 malformedLines
        if (meta.malformedLines && meta.malformedLines > 0) {
          store.showToast(
            'warn',
            `Bundle 中有 ${meta.malformedLines} 行解析失败，可能是写入不完整`
          )
        }

        store.showToast('info', `已加载 Bundle: ${result.ref.label}`)
        setLoading(false)
      } catch (err) {
        const msg = err instanceof Error ? err.message : '未知错误'
        setError(msg)
        store.setLoading(null, msg)
        store.showToast('error', `加载 Bundle 失败: ${msg}`)
        setLoading(false)
      }
    },
    [store, queryClient]
  )

  // 打开目录选择器
  const openDirectory = useCallback(async () => {
    if (!supportsDirectoryPicker()) {
      store.showToast(
        'error',
        '当前浏览器不支持目录选择 (File System Access API)，请用 Chrome/Edge 95+，或选择 fixtures'
      )
      return
    }

    try {
      const result = await pickBundleDirectory()
      if (!result) {
        // 用户取消
        return
      }
      await loadBundle(result)
    } catch (err) {
      const msg = err instanceof Error ? err.message : '未知错误'
      setError(msg)
      store.showToast('error', `目录选择失败: ${msg}`)
    }
  }, [loadBundle, store])

  // 从 fixtures 加载
  const loadFixtureById = useCallback(
    async (fixtureId: string) => {
      try {
        const fixture = fixtures.find((f) => f.id === fixtureId)
        if (!fixture) {
          throw new Error(`Fixture not found: ${fixtureId}`)
        }

        const result = await loadFromUrl(fixture.path, fixture.id)
        await loadBundle(result)
      } catch (err) {
        const msg = err instanceof Error ? err.message : '未知错误'
        setError(msg)
        store.showToast('error', `加载 Fixture 失败: ${msg}`)
      }
    },
    [fixtures, loadBundle, store]
  )

  // 设置为 Baseline
  const setAsBaseline = useCallback(
    (bundleId: string) => {
      // 切换前清理旧 bundle（§18.3）
      const workspace = store.activeWorkspace()
      if (workspace?.baselineBundleId) {
        queryClient.unload(workspace.baselineBundleId)
      }

      store.bindBaseline(bundleId)
      store.showToast('info', `已设置 Baseline: ${bundleId}`)
    },
    [store, queryClient]
  )

  return {
    fixtures,
    loading,
    error,
    openDirectory,
    loadFixtureById,
    setAsBaseline,
  }
}
