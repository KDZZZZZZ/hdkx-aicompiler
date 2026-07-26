/**
 * Bundle 读取（需求 §2.1、§19 本地优先）。
 *
 * 两条来源：
 * 1. 本地目录 —— File System Access API 的 showDirectoryPicker，文件不出浏览器进程。
 * 2. fixtures URL —— dev 环境下从 publicDir 直接 fetch，用于样例与真实产物回归。
 *
 * 两条路径都只在主线程做 IO，读完把文本交给 Worker，Worker 不碰文件系统。
 */

import { BUNDLE_FILES } from './contract'
import type { LoadPayload } from './query-protocol'

export interface BundleRef {
  id: string
  label: string
  /** 来源说明，展示在顶部栏，让用户随时知道数据从哪来（§19.3）。 */
  hint: string
  kind: 'directory' | 'url'
}

export interface BundleLoadResult {
  ref: BundleRef
  payload: LoadPayload
}

/** IR artifact 通常较大且按需查看，首次加载只索引路径不读内容。 */
const EAGER_ARTIFACT_LIMIT = 64

// ---------------------------------------------------------------------------
// 目录来源
// ---------------------------------------------------------------------------

interface FsDirectoryHandle {
  name: string
  kind: 'directory'
  values(): AsyncIterableIterator<FsDirectoryHandle | FsFileHandle>
  getFileHandle(name: string): Promise<FsFileHandle>
  getDirectoryHandle(name: string): Promise<FsDirectoryHandle>
}
interface FsFileHandle {
  name: string
  kind: 'file'
  getFile(): Promise<File>
}

export function supportsDirectoryPicker(): boolean {
  return typeof (globalThis as { showDirectoryPicker?: unknown }).showDirectoryPicker === 'function'
}

export async function pickBundleDirectory(): Promise<BundleLoadResult | null> {
  const picker = (globalThis as { showDirectoryPicker?: () => Promise<FsDirectoryHandle> })
    .showDirectoryPicker
  if (!picker) throw new Error('当前浏览器不支持目录选择，请改用 fixtures 或拖入 bundle 目录')
  let dir: FsDirectoryHandle
  try {
    dir = await picker()
  } catch {
    return null // 用户取消
  }
  return loadFromDirectoryHandle(dir)
}

export async function loadFromDirectoryHandle(dir: FsDirectoryHandle): Promise<BundleLoadResult> {
  const readText = async (name: string): Promise<string | null> => {
    try {
      const fh = await dir.getFileHandle(name)
      return await (await fh.getFile()).text()
    } catch {
      return null
    }
  }

  const events = await readText(BUNDLE_FILES.events)
  if (events == null) {
    throw new Error(`目录 "${dir.name}" 里没有 ${BUNDLE_FILES.events}，不是一个 KXC bundle`)
  }

  const artifacts: Record<string, string> = {}
  try {
    const artifactDir = await dir.getDirectoryHandle(BUNDLE_FILES.artifactsDir)
    await collectArtifacts(artifactDir, BUNDLE_FILES.artifactsDir, artifacts, EAGER_ARTIFACT_LIMIT)
  } catch {
    // 没有 artifacts 目录是正常的
  }

  return {
    ref: {
      id: `dir:${dir.name}`,
      label: dir.name,
      hint: '本地目录（数据未离开本机）',
      kind: 'directory',
    },
    payload: {
      manifest: await readText(BUNDLE_FILES.manifest),
      events,
      summary: await readText(BUNDLE_FILES.summary),
      diagnosis: await readText(BUNDLE_FILES.diagnosisJson),
      trace: await readText(BUNDLE_FILES.trace),
      artifacts,
    },
  }
}

async function collectArtifacts(
  dir: FsDirectoryHandle,
  prefix: string,
  out: Record<string, string>,
  budget: number,
): Promise<number> {
  let left = budget
  for await (const entry of dir.values()) {
    if (left <= 0) break
    const path = `${prefix}/${entry.name}`
    if (entry.kind === 'directory') {
      left = await collectArtifacts(entry as FsDirectoryHandle, path, out, left)
    } else {
      out[path] = await (await (entry as FsFileHandle).getFile()).text()
      left -= 1
    }
  }
  return left
}

// ---------------------------------------------------------------------------
// URL 来源（fixtures / 静态托管）
// ---------------------------------------------------------------------------

export interface FixtureIndexEntry {
  id: string
  path: string
  variant: string
  event_count: number
}

export async function listFixtures(): Promise<FixtureIndexEntry[]> {
  try {
    const res = await fetch('/bundles/index.json')
    if (!res.ok) return []
    const json = (await res.json()) as { bundles?: FixtureIndexEntry[] }
    return json.bundles ?? []
  } catch {
    return []
  }
}

export async function loadFromUrl(baseUrl: string, label: string): Promise<BundleLoadResult> {
  const base = baseUrl.replace(/\/$/, '')
  const fetchText = async (name: string): Promise<string | null> => {
    try {
      const res = await fetch(`${base}/${name}`)
      return res.ok ? await res.text() : null
    } catch {
      return null
    }
  }

  const events = await fetchText(BUNDLE_FILES.events)
  if (events == null) throw new Error(`${base} 下没有 ${BUNDLE_FILES.events}`)

  // 静态托管无法列目录，artifact 依据事件里的 pass 信息按需拉取，这里先留空。
  return {
    ref: {
      id: `url:${base}`,
      label,
      hint: `本地开发服务 ${base}`,
      kind: 'url',
    },
    payload: {
      manifest: await fetchText(BUNDLE_FILES.manifest),
      events,
      summary: await fetchText(BUNDLE_FILES.summary),
      diagnosis: await fetchText(BUNDLE_FILES.diagnosisJson),
      trace: await fetchText(BUNDLE_FILES.trace),
      artifacts: {},
    },
  }
}

/** 按需取单个 artifact（IR Tile 用）。URL 来源才需要，目录来源已在加载时读入。 */
export async function fetchArtifact(baseUrl: string, relPath: string): Promise<string | null> {
  try {
    const res = await fetch(`${baseUrl.replace(/\/$/, '')}/${relPath}`)
    return res.ok ? await res.text() : null
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// 脱敏（§19.5）
// ---------------------------------------------------------------------------

/**
 * 分享前移除本地路径、用户名与主机名。
 * 只处理已知会带这些信息的字段，不做全文正则替换，避免误伤 IR 内容。
 */
export function sanitizeForShare<T extends Record<string, unknown>>(obj: T): T {
  const clone = JSON.parse(JSON.stringify(obj)) as Record<string, unknown>
  const scrub = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(scrub)
    if (node && typeof node === 'object') {
      const o = node as Record<string, unknown>
      for (const key of Object.keys(o)) {
        if (key === 'bundle_dir' || key === 'hint' || key === 'path') {
          if (typeof o[key] === 'string') o[key] = '<redacted>'
        } else {
          o[key] = scrub(o[key])
        }
      }
      return o
    }
    return node
  }
  return scrub(clone) as T
}
