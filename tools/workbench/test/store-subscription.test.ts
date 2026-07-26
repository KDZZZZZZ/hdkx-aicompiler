/**
 * 订阅方式的静态守卫。
 *
 * 「无限重渲染」在这个项目里已经出现过两次，根因都是同一种写法。它的代价很高：
 * 类型检查发现不了，构建也照样通过，只有真的在浏览器里打开才会白屏。
 * 所以这里直接在源码层面把这两种写法钉死，而不是指望下次 review 能看出来。
 */

import { describe, it, expect } from 'vitest'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const SRC = join(__dirname, '..', 'src')

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (/\.tsx?$/.test(p)) out.push(p)
  }
  return out
}

/** 去掉注释，避免把解释这条规则的注释本身判成违规。 */
function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '')
}

const files = walk(SRC).map((p) => ({ path: relative(SRC, p), code: stripComments(readFileSync(p, 'utf8')) }))

describe('store 订阅方式', () => {
  it('不允许 useWorkbench() 不带 selector 订阅整个 store', () => {
    // 每次 set() 都会换掉整个 state 引用，订阅它等于"任何状态变化都重渲染"，
    // 一旦组件里还有 effect + setState，就会构成死循环。
    // 需要在回调里调 action 时，用 useWorkbench.getState()。
    const bad = files.filter((f) => /useWorkbench\(\s*\)/.test(f.code))
    expect(bad.map((f) => f.path)).toEqual([])
  })

  it('不允许在 selector 里调用返回新对象的派生函数', () => {
    // globalContext() / columnsOf() / tilesOf() 每次都返回新对象或新数组，
    // zustand 用 Object.is 比较快照，新引用会被判定为"状态又变了"。
    // 对应的安全写法是 useGlobalContext() / useColumnTiles()。
    const bad = files.filter((f) =>
      /useWorkbench\(\s*\(\s*s\s*\)\s*=>\s*s\.(globalContext|columnsOf|tilesOf)\s*\(/.test(f.code),
    )
    expect(bad.map((f) => f.path)).toEqual([])
  })

  it('不允许 selector 直接返回对象字面量或 map/filter 结果', () => {
    const bad = files.filter((f) =>
      /useWorkbench\(\s*\(\s*s\s*\)\s*=>\s*(\{[^}]*:|.*\.(map|filter)\s*\()/.test(f.code),
    )
    expect(bad.map((f) => f.path)).toEqual([])
  })
})
