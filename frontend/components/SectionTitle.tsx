import { ReactNode } from 'react'

interface SectionTitleProps {
  children: ReactNode
  subtitle?: string
  className?: string
}

export default function SectionTitle({ children, subtitle, className = '' }: SectionTitleProps) {
  return (
    <div className={className}>
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{children}</h2>
      {subtitle && <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{subtitle}</p>}
    </div>
  )
}

