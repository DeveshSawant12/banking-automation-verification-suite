import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ShieldCheck } from 'lucide-react'
import { login as apiLogin } from '../api/client'
import { useAuth } from '../auth/AuthContext'

const ROLE_DEFAULT_ROUTES = {
  ADMIN: '/dashboard',
  BANK_STAFF: '/dashboard',
  CUSTOMER: '/onboarding',
}

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const res = await apiLogin(email, password)
      const { access_token, user } = res.data
      login(access_token, user)
      navigate(ROLE_DEFAULT_ROUTES[user.role] || '/dashboard', { replace: true })
    } catch (err) {
      setError(
        err.response?.data?.detail || 'Invalid credentials. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-surface)]">
      <div className="w-full max-w-sm">
        {/* Brand */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 rounded-xl bg-[var(--color-ink)] flex items-center justify-center mb-3">
            <ShieldCheck className="w-6 h-6 text-[var(--color-accent)]" />
          </div>
          <h1 className="text-xl font-bold text-[var(--color-ink)]">BankVerify KYC Suite</h1>
          <p className="text-sm text-[var(--color-muted)] mt-0.5">Sign in to your account</p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl border border-[var(--color-border)] shadow-sm p-8">
          {error && (
            <div className="mb-5 px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-xs font-semibold text-[var(--color-ink)] mb-1.5">
                Email address
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@bank.com"
                className="w-full rounded-lg border border-[var(--color-border)] px-3.5 py-2.5 text-sm
                           outline-none focus:border-[var(--color-primary)] focus:ring-2
                           focus:ring-[var(--color-primary)]/20 transition-colors"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--color-ink)] mb-1.5">
                Password
              </label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded-lg border border-[var(--color-border)] px-3.5 py-2.5 text-sm
                           outline-none focus:border-[var(--color-primary)] focus:ring-2
                           focus:ring-[var(--color-primary)]/20 transition-colors"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)]
                         disabled:opacity-60 disabled:cursor-not-allowed
                         text-white text-sm font-semibold py-2.5 rounded-lg transition-colors"
            >
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
