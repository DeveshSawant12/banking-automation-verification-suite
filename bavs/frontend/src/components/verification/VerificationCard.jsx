import Badge from '../ui/Badge'
import { CheckCircle, XCircle, AlertTriangle, HelpCircle } from 'lucide-react'

const ICON_MAP = {
  REAL: CheckCircle,
  LIVE: CheckCircle,
  MATCH: CheckCircle,
  APPROVED: CheckCircle,
  TAMPERED: XCircle,
  SPOOF: XCircle,
  MISMATCH: XCircle,
  REJECTED: XCircle,
  INCONCLUSIVE: AlertTriangle,
  REVIEW_REQUIRED: AlertTriangle,
}

export default function VerificationCard({ title, verdict, details = [] }) {
  const Icon = ICON_MAP[verdict] || HelpCircle

  const iconColor = {
    REAL: 'text-emerald-500',
    LIVE: 'text-emerald-500',
    MATCH: 'text-emerald-500',
    APPROVED: 'text-emerald-500',
    TAMPERED: 'text-red-500',
    SPOOF: 'text-red-500',
    MISMATCH: 'text-red-500',
    REJECTED: 'text-red-500',
    INCONCLUSIVE: 'text-amber-500',
    REVIEW_REQUIRED: 'text-amber-500',
  }[verdict] || 'text-gray-400'

  return (
    <div className="bg-white rounded-xl border border-[var(--color-border)] p-5 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <Icon className={`w-5 h-5 shrink-0 ${iconColor}`} />
          <span className="text-sm font-semibold text-[var(--color-ink)]">{title}</span>
        </div>
        <Badge label={verdict} />
      </div>

      {details.length > 0 && (
        <ul className="space-y-1 pt-1 border-t border-[var(--color-border)]">
          {details.map(({ label, value }) => (
            <li key={label} className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-muted)]">{label}</span>
              <span className="font-medium text-[var(--color-ink)]">{value}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
