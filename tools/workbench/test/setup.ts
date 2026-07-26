/**
 * 测试环境补齐。
 *
 * jsdom 没有实现 ResizeObserver / IntersectionObserver / matchMedia，
 * 而工作台的横向虚拟化、Content Fit 测量、Overview 渲染预算都依赖它们。
 * 这里给出最小可用替身：只保证组件能挂载并调用，不模拟真实的尺寸回调
 * （尺寸相关行为在真实浏览器里验证，不在 jsdom 里假装）。
 */

class NoopObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return []
  }
}

globalThis.ResizeObserver ??= NoopObserver as unknown as typeof ResizeObserver
globalThis.IntersectionObserver ??= NoopObserver as unknown as typeof IntersectionObserver

/**
 * jsdom 没有 Worker。查询层在模块初始化时就会 new Worker()，缺了它任何
 * 渲染到顶栏的测试都会直接抛错。
 * 这里用直接赋值而不是 vi.stubGlobal —— 后者会被 vi.unstubAllGlobals() 清掉，
 * 于是"先跑的用例把后跑的用例搞崩"。查询逻辑本身由 query-pipeline.test.ts 覆盖。
 */
class NoopWorker {
  onmessage: unknown = null
  onerror: unknown = null
  postMessage() {}
  terminate() {}
  addEventListener() {}
  removeEventListener() {}
}
globalThis.Worker ??= NoopWorker as unknown as typeof Worker

// React 18 要求显式声明处于 act 环境，否则 act() 会打警告并可能吞掉更新。
;(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true

if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia
}

// jsdom 的 Element 缺少这两个滚动方法：StripView 定位聚焦列、
// 命令面板把选中项滚进视野时都会调用。
if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = function scrollTo() {}
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function scrollIntoView() {}
}
