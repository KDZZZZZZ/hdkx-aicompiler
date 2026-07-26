/**
 * 行级 diff（真 LCS）。
 *
 * 上一版只同步公共前缀，遇到第一处差异就把两侧剩余部分整段标成删除+新增——
 * 顶部改一行，下面几百行未变的 IR 全被标红标绿，diff 视图完全误导
 * （Codex review 抓到，注释里"极简 LCS"其实自认了）。
 *
 * 这里做标准动态规划 LCS，外加两道工程保护：
 * 1. 先剥公共前缀/后缀——IR 前后版本大多只改中间一小段，剥完后 DP 规模骤降；
 * 2. 中段规模超过上限时退化为整段替换（保留已剥出的前后缀），
 *    避免病态输入把浏览器算挂。退化路径的行为与旧实现相同，但只发生在
 *    真正巨大的 diff 上，而不是所有 diff 上。
 */

export interface DiffLine {
  line: string
  /** -1=删除，0=未变，1=新增 */
  status: -1 | 0 | 1
}

/** 中段 DP 的最大单元数。9e6 单元在现代机器上 <50ms，够覆盖数千行的 IR。 */
const MAX_DP_CELLS = 9_000_000

export function lcsDiff(before: string[], after: string[]): DiffLine[] {
  // 剥公共前缀
  let start = 0
  while (start < before.length && start < after.length && before[start] === after[start]) {
    start += 1
  }
  // 剥公共后缀（注意不越过前缀）
  let endB = before.length
  let endA = after.length
  while (endB > start && endA > start && before[endB - 1] === after[endA - 1]) {
    endB -= 1
    endA -= 1
  }

  const head: DiffLine[] = before.slice(0, start).map((line) => ({ line, status: 0 as const }))
  const tail: DiffLine[] = before.slice(endB).map((line) => ({ line, status: 0 as const }))
  const midB = before.slice(start, endB)
  const midA = after.slice(start, endA)

  return [...head, ...diffMiddle(midB, midA), ...tail]
}

function diffMiddle(b: string[], a: string[]): DiffLine[] {
  if (b.length === 0 && a.length === 0) return []
  if (b.length === 0) return a.map((line) => ({ line, status: 1 as const }))
  if (a.length === 0) return b.map((line) => ({ line, status: -1 as const }))

  // 规模保护：退化为整段替换。前后缀已剥掉，只有中段真的巨大才会走到这里。
  if (b.length * a.length > MAX_DP_CELLS) {
    return [
      ...b.map((line) => ({ line, status: -1 as const })),
      ...a.map((line) => ({ line, status: 1 as const })),
    ]
  }

  // 标准 LCS DP。dp[i][j] = b[i:] 与 a[j:] 的最长公共子序列长度。
  // 用一维滚动数组省内存：dp 逆序填表。
  const cols = a.length + 1
  const dp = new Uint32Array((b.length + 1) * cols)
  for (let i = b.length - 1; i >= 0; i -= 1) {
    for (let j = a.length - 1; j >= 0; j -= 1) {
      dp[i * cols + j] =
        b[i] === a[j]
          ? (dp[(i + 1) * cols + j + 1] ?? 0) + 1
          : Math.max(dp[(i + 1) * cols + j] ?? 0, dp[i * cols + j + 1] ?? 0)
    }
  }

  // 回溯
  const out: DiffLine[] = []
  let i = 0
  let j = 0
  while (i < b.length && j < a.length) {
    if (b[i] === a[j]) {
      out.push({ line: b[i] ?? '', status: 0 })
      i += 1
      j += 1
    } else if ((dp[(i + 1) * cols + j] ?? 0) >= (dp[i * cols + j + 1] ?? 0)) {
      out.push({ line: b[i] ?? '', status: -1 })
      i += 1
    } else {
      out.push({ line: a[j] ?? '', status: 1 })
      j += 1
    }
  }
  while (i < b.length) {
    out.push({ line: b[i] ?? '', status: -1 })
    i += 1
  }
  while (j < a.length) {
    out.push({ line: a[j] ?? '', status: 1 })
    j += 1
  }
  return out
}
