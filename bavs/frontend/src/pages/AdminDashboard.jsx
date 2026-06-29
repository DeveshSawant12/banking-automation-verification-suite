import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Users, CheckCircle, AlertTriangle, XCircle, RefreshCw } from 'lucide-react'
import { listCases, getAdminSummary } from '../api/client'
import { Card, Spinner, EmptyState } from '../components/ui/Card'
import Badge from '../components/ui/Badge'

function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="bg-white rounded-xl border border-[var(--color-border)] p-5 flex items-center gap-4">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${color}`}>
        <Icon className="w-5 h-5 text-white" />
      </div>
      <div>
        <p className="text-2xl font-bold text-[var(--color-ink)]">{value ?? '—'}</p>
        <p className="text-xs text-[var(--color-muted)] mt-0.5">{label}</p>
      </div>
    </div>
  )
}

const STATUS_FILTER_OPTIONS = ['ALL', 'PENDING', 'PROCESSING', 'APPROVED', 'REVIEW_REQUIRED', 'REJECTED']
const RISK_FILTER_OPTIONS = ['ALL', 'LOW', 'MEDIUM', 'HIGH']

export default function AdminDashboard() {
  const navigate = useNavigate()
  const [cases, setCases] = useState([])
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState('ALL')
  const [riskFilter, setRiskFilter] = useState('ALL')
  const [page, setPage] = useState(0)
  const PAGE_SIZE = 20

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const params = {
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }
      if (statusFilter !== 'ALL') params.status = statusFilter

      const [casesRes, summaryRes] = await Promise.all([
        listCases(params),
        getAdminSummary(),
      ])
      setCases(casesRes.data)
      setSummary(summaryRes.data)
    } catch {
      // errors are visible in the empty state
    } finally {
      setLoading(false)
    }
  }, [statusFilter, page])

  useEffect(() => { fetchData() }, [fetchData])

  const filtered = riskFilter === 'ALL'
    ? cases
    : cases.filter((c) => c.fraud_risk?.risk_level === riskFilter)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-ink)]">KYC Dashboard</h1>
          <p className="text-sm text-[var(--color-muted)] mt-0.5">
            All customer onboarding cases
          </p>
        </div>
        <button
          onClick={fetchData}
          className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-primary)]
                     hover:text-[var(--color-primary-hover)] transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </button>
      </div>

      {/* Stats */}
      {summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            icon={Users}
            label="Total Cases"
            value={summary.total_cases}
            color="bg-[var(--color-primary)]"
          />
          <StatCard
            icon={CheckCircle}
            label="Approved"
            value={summary.approved}
            color="bg-[var(--color-accent)]"
          />
          <StatCard
            icon={AlertTriangle}
            label="Under Review"
            value={summary.review_required}
            color="bg-amber-500"
          />
          <StatCard
            icon={XCircle}
            label="Rejected"
            value={summary.rejected}
            color="bg-[var(--color-danger)]"
          />
        </div>
      )}

      {/* Filters */}
      <Card title="Cases">
        <div className="flex flex-wrap gap-3 mb-5">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[var(--color-muted)] font-medium">Status:</span>
            <div className="flex flex-wrap gap-1">
              {STATUS_FILTER_OPTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => { setStatusFilter(s); setPage(0) }}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                    statusFilter === s
                      ? 'bg-[var(--color-ink)] text-white'
                      : 'bg-[var(--color-surface)] text-[var(--color-muted)] hover:bg-[var(--color-border)]'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-[var(--color-muted)] font-medium">Risk:</span>
            <div className="flex gap-1">
              {RISK_FILTER_OPTIONS.map((r) => (
                <button
                  key={r}
                  onClick={() => setRiskFilter(r)}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                    riskFilter === r
                      ? 'bg-[var(--color-ink)] text-white'
                      : 'bg-[var(--color-surface)] text-[var(--color-muted)] hover:bg-[var(--color-border)]'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div className="flex justify-center py-10"><Spinner /></div>
        ) : filtered.length === 0 ? (
          <EmptyState
            icon={Users}
            title="No cases found"
            description="Adjust your filters or wait for new onboarding submissions."
          />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--color-border)]">
                    {['Case ID', 'Customer', 'Status', 'Risk Level', 'Risk Score', 'Submitted'].map(
                      (h) => (
                        <th
                          key={h}
                          className="text-left text-xs font-semibold text-[var(--color-muted)] uppercase tracking-wider pb-3 pr-4"
                        >
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--color-border)]">
                  {filtered.map((c) => (
                    <tr
                      key={c.id}
                      onClick={() => navigate(`/cases/${c.id}`)}
                      className="cursor-pointer hover:bg-[var(--color-surface)] transition-colors"
                    >
                      <td className="py-3 pr-4">
                        <span className="font-mono text-xs text-[var(--color-muted)]">
                          {c.id.slice(0, 8)}…
                        </span>
                      </td>
                      <td className="py-3 pr-4 font-medium text-[var(--color-ink)]">
                        {c.customer_name || c.customer_id?.slice(0, 8) + '…'}
                      </td>
                      <td className="py-3 pr-4">
                        <Badge label={c.status} />
                      </td>
                      <td className="py-3 pr-4">
                        {c.fraud_risk ? (
                          <Badge label={c.fraud_risk.risk_level} />
                        ) : (
                          <span className="text-xs text-[var(--color-muted)]">Pending</span>
                        )}
                      </td>
                      <td className="py-3 pr-4 font-semibold text-[var(--color-ink)]">
                        {c.fraud_risk?.total_score ?? '—'}
                      </td>
                      <td className="py-3 pr-4 text-[var(--color-muted)]">
                        {new Date(c.created_at).toLocaleDateString('en-IN', {
                          day: '2-digit', month: 'short', year: 'numeric',
                        })}
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
              <span className="text-xs text-[var(--color-muted)]">Page {page + 1}</span>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={filtered.length < PAGE_SIZE}
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
