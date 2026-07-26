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

// jsdom 的 Element 没有 scrollTo，StripView 定位聚焦列时会调用。
if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = function scrollTo() {}
}
