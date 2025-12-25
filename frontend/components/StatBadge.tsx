interface StatBadgeProps {
  label: string
  value: string | number
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  size?: 'sm' | 'md' | 'lg'
}

export default function StatBadge({ label, value, variant = 'default', size = 'md' }: StatBadgeProps) {
  const variantStyles = {
    default: 'bg-gray-100 text-gray-800 border-gray-200',
    success: 'bg-green-100 text-green-800 border-green-200',
    warning: 'bg-amber-100 text-amber-800 border-amber-200',
    danger: 'bg-red-100 text-red-800 border-red-200',
    info: 'bg-blue-100 text-blue-800 border-blue-200',
  }

  const sizeStyles = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1.5',
    lg: 'text-base px-4 py-2',
  }

  return (
    <div className={`inline-flex items-center gap-2 rounded-full border font-medium ${variantStyles[variant]} ${sizeStyles[size]}`}>
      <span className="font-semibold">{value}</span>
      <span className="opacity-70">{label}</span>
    </div>
  )
}

