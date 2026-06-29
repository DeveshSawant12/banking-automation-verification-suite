import { useState, useEffect, useCallback } from 'react'
import { ClipboardList, RefreshCw } from 'lucide-react'
import { getAuditLogs } from '../api/client'
import { Card, Spinner, EmptyState } from '../components/ui/Card'
import Badge from '../components/ui/Badge'

const EVENT_COLORS = {
  OCR_EXTRACTION_COMPLETED: 'text-blue-600 bg-blue-50',
  AADHAAR_TAMPERING_CHECK_COMPLETED: 'text-purple-600 bg-purple-50',
  PAN_TAMPERING_CHECK_COMPLETED: 'text-purple-600 bg-purple-50',
  FACE_VERIFICATION_COMPLETED: 'text-indigo-600 bg-indigo-50',
  CROSS_DOCUMENT_VERIFICATION_COMPLETED: 'text-teal-600 bg-teal-50',
  LIVENESS_CHECK_COMPLETED: 'text-cyan-600 bg-cyan-50',
  FRAUD_RISK_SCORED: 'text-amber-600 bg-amber-50',
  EXPLANATION_GENERATED: 'text-orange-600 bg-orange-50',
  KYC_CASE_CREATED: 'text-green-600 bg-green-50',
  KYC_CASE_DECISION_MADE: 'text-emerald-600 bg-emerald-50',
  DOCUMENT_UPLOADED: 'text-slate-600 bg-slate-50',
}

function EventTypePill({ type }) {
  const colors = EVENT_COLORS[type] || 'text-gray-600 bg-gray-50'
  const label = type.replace(/_/g, ' ')
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${colors}`}>
      {label}
    </span>
  )
}

const PAGE_SIZE = 50

export default function AuditLogs() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)

  const fetchLogs = useCallback(async () => {
    setLoading(true)
    try {
      const res = await getAuditLogs({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      })
      setLogs(res.data)
    } catch {
      // errors visible via empty state
    } finally {
      setLoading(false)
    }
  }, [page])

  useEffect(() => { fetchLogs() }, [fetchLogs])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-ink)]">Audit Logs</h1>
          <p className="text-sm text-[var(--color-muted)] mt-0.5">
            Immutable, append-only verification event trail
          </p>
        </div>
        <button
          onClick={fetchLogs}
          className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-primary)]
                     hover:text-[var(--color-primary-hover)] transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </button>
      </div>

      <Card title={`Audit Events (page ${page + 1})`}>
        {loading ? (
          <div className="flex justify-center py-10"><Spinner /></div>
        ) : logs.length === 0 ? (
          <EmptyState
            icon={ClipboardList}
            title="No audit entries yet"
            description="Events are written here automatically as the verification pipeline runs."
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--color-border)]">
                    {['Timestamp', 'Event', 'Case ID', 'Status', 'Risk Score', 'Decision'].map(
                      (h) => (
                        <th
                          key={h}
                          className="text-left text-xs font-semibold text-[var(--color-muted)]
                                     uppercase tracking-wider pb-3 pr-4 whitespace-nowrap"
                        >
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--color-border)]">
                  {logs.map((log) => (
                    <tr key={log.id} className="hover:bg-[var(--color-surface)] transition-colors">
                      <td className="py-3 pr-4 text-xs text-[var(--color-muted)] whitespace-nowrap">
                        {new Date(log.timestamp).toLocaleString('en-IN', {
                          day: '2-digit', month: 'short', year: 'numeric',
                          hour: '2-digit', minute: '2-digit', second: '2-digit',
                        })}
                      </td>
                      <td className="py-3 pr-4">
                        <EventTypePill type={log.event_type} />
                      </td>
                      <td className="py-3 pr-4">
                        {log.kyc_case_id ? (
                          <span className="font-mono text-xs text-[var(--color-muted)]">
                            {log.kyc_case_id.slice(0, 8)}…
                          </span>
                        ) : (
                          <span className="text-xs text-[var(--color-muted)]">—</span>
                        )}
                      </td>
                      <td className="py-3 pr-4">
                        {log.verification_status ? (
                          <Badge label={log.verification_status} />
                        ) : (
                          <span className="text-xs text-[var(--color-muted)]">—</span>
                        )}
                      </td>
                      <td className="py-3 pr-4 font-semibold text-[var(--color-ink)]">
                        {log.risk_score ?? '—'}
                      </td>
                      <td className="py-3 pr-4 text-xs text-[var(--color-muted)]">
                        {log.decision ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between pt-4 mt-2 border-t border-[var(--color-border)]">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="text-xs font-medium text-[var(--color-primary)] disabled:opacity-40
                           hover:text-[var(--color-primary-hover)] transition-colors"
              >
                ← Previous
              </button>
              <span className="text-xs text-[var(--color-muted)]">
                Showing {page * PAGE_SIZE + 1}–{page * PAGE_SIZE + logs.length}
              </span>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={logs.length < PAGE_SIZE}
                className="text-xs font-medium text-[var(--color-primary)] disabled:opacity-40
                           hover:text-[var(--color-primary-hover)] transition-colors"
              >
                Next →
              </button>
            </div>
          </>
        )}
      </Card>
    </div>
  )
}
