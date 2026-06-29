import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import ProtectedRoute from './auth/ProtectedRoute'
import Layout from './components/layout/Layout'

import Login from './pages/Login'
import AdminDashboard from './pages/AdminDashboard'
import CustomerOnboarding from './pages/CustomerOnboarding'
import CaseDetail from './pages/CaseDetail'
import AuditLogs from './pages/AuditLogs'
import ChatbotPage from './pages/ChatbotPage'

function Unauthorized() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-surface)]">
      <div className="text-center">
        <p className="text-2xl font-bold text-[var(--color-ink)] mb-2">Access Denied</p>
        <p className="text-sm text-[var(--color-muted)]">
          You do not have permission to view this page.
        </p>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/unauthorized" element={<Unauthorized />} />

          {/* Admin + Bank Staff: case list + stats dashboard */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute allowedRoles={['ADMIN', 'BANK_STAFF']}>
                <Layout><AdminDashboard /></Layout>
              </ProtectedRoute>
            }
          />

          {/* Case detail: accessible to ADMIN, BANK_STAFF, and the owning CUSTOMER */}
          <Route
            path="/cases/:caseId"
            element={
              <ProtectedRoute>
                <Layout><CaseDetail /></Layout>
              </ProtectedRoute>
            }
          />

          {/* Customer self-service onboarding */}
          <Route
            path="/onboarding"
            element={
              <ProtectedRoute allowedRoles={['CUSTOMER']}>
                <Layout><CustomerOnboarding /></Layout>
              </ProtectedRoute>
            }
          />

          {/* Customer: their own verification status */}
          <Route
            path="/verification"
            element={
              <ProtectedRoute allowedRoles={['CUSTOMER']}>
                <Layout>
                  <Navigate to="/dashboard" replace />
                </Layout>
              </ProtectedRoute>
            }
          />

          {/* Audit logs: admin only */}
          <Route
            path="/audit"
            element={
              <ProtectedRoute allowedRoles={['ADMIN']}>
                <Layout><AuditLogs /></Layout>
              </ProtectedRoute>
            }
          />

          {/* Chatbot: any authenticated user */}
          <Route
            path="/chatbot"
            element={
              <ProtectedRoute>
                <Layout><ChatbotPage /></Layout>
              </ProtectedRoute>
            }
          />

          {/* Default redirect based on role is handled post-login in Login.jsx */}
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
