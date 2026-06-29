export function Card({ title, children, className = '' }) {
  return (
    <div className={`bg-white rounded-xl border border-[var(--color-border)] shadow-sm ${className}`}>
      {title && (
        <div className="px-6 py-4 border-b border-[var(--color-border)]">
          <h3 className="text-sm font-semibold text-[var(--color-ink)] uppercase tracking-wider">
            {title}
          </h3>
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  )
}

export function Spinner({ size = 'md' }) {
  const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-10 h-10' }
  return (
    <div
      className={`${sizes[size]} border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin`}
    />
  )
}

export function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {Icon && <Icon className="w-12 h-12 text-[var(--color-border)] mb-4" />}
      <p className="text-base font-medium text-[var(--color-ink)]">{title}</p>
      {description && (
        <p className="mt-1 text-sm text-[var(--color-muted)]">{description}</p>
      )}
    </div>
  )
}
