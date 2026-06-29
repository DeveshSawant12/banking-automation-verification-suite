import { Navigate } from 'react-router-dom'
import { useAuth } from './AuthContext'

/**
 * Wraps a route to require authentication, and optionally a specific set
 * of roles. Unauthenticated users go to /login; authenticated users
 * without the required role go to /unauthorized.
 *
 * Usage:
 *   <ProtectedRoute allowedRoles={['ADMIN', 'BANK_STAFF']}>
 *     <AuditLogs />
 *   </ProtectedRoute>
 */
export default function ProtectedRoute({ children, allowedRoles }) {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--color-surface)]">
        <div className="w-8 h-8 border-4 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!user) return <Navigate to="/login" replace />

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    return <Navigate to="/unauthorized" replace />
  }

  return children
}
