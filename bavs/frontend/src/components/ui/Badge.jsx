const VARIANTS = {
  // Tampering verdicts
  REAL: 'bg-emerald-100 text-emerald-800',
  TAMPERED: 'bg-red-100 text-red-800',
  INCONCLUSIVE: 'bg-amber-100 text-amber-800',
  // Risk levels
  LOW: 'bg-emerald-100 text-emerald-800',
  MEDIUM: 'bg-amber-100 text-amber-800',
  HIGH: 'bg-red-100 text-red-800',
  // Liveness
  LIVE: 'bg-emerald-100 text-emerald-800',
  SPOOF: 'bg-red-100 text-red-800',
  // Case status
  PENDING: 'bg-gray-100 text-gray-700',
  PROCESSING: 'bg-blue-100 text-blue-800',
  APPROVED: 'bg-emerald-100 text-emerald-800',
  REVIEW_REQUIRED: 'bg-amber-100 text-amber-800',
  REJECTED: 'bg-red-100 text-red-800',
  // Generic
  MATCH: 'bg-emerald-100 text-emerald-800',
  MISMATCH: 'bg-red-100 text-red-800',
  default: 'bg-gray-100 text-gray-700',
}

export default function Badge({ label, className = '' }) {
  const variant = VARIANTS[label] || VARIANTS.default
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold tracking-wide ${variant} ${className}`}
    >
      {label}
    </span>
  )
}
