import { ReactNode } from 'react'

interface GlassCardProps {
  children: ReactNode
  className?: string
}

export default function GlassCard({ children, className = '' }: GlassCardProps) {
  return (
    <div className={`backdrop-blur-xl bg-white/80 dark:bg-gray-900/80 border border-white/20 rounded-2xl shadow-xl ${className}`}>
      {children}
    </div>
  )
}

