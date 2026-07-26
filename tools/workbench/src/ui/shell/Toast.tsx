import { useEffect } from 'react'
import { useWorkbench } from '../../state/store'
import './shell.css'

/**
 * 吐司通知组件（需求 §5.1、§21）。
 * 在右下角显示临时通知，支持 error / warn / info 三种级别。
 * 自动消失或用户可手动关闭。
 */
export function Toast(): JSX.Element {
  const toast = useWorkbench((s) => s.toast)
  const showToast = useWorkbench((s) => s.showToast)

  useEffect(() => {
    if (!toast) return
    // 自动消失（3 秒）
    const t = setTimeout(() => showToast(toast.kind, ''), 3000)
    return () => clearTimeout(t)
  }, [toast, showToast])

  if (!toast) return <></>

  const getIcon = () => {
    switch (toast.kind) {
      case 'error':
        return '✕'
      case 'warn':
        return '⚠'
      case 'info':
        return 'ⓘ'
    }
  }

  return (
    <div className="toast-container">
      <div className={`toast ${toast.kind}`}>
        <span>{getIcon()}</span>
        <span>{toast.text}</span>
        <button
          className="toast-close"
          onClick={() => showToast(toast.kind, '')}
          aria-label="关闭通知"
        >
          ✕
        </button>
      </div>
    </div>
  )
}
