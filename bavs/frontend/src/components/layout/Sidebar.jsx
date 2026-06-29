import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  CheckCircle,
  ClipboardList,
  MessageSquare,
  LogOut,
  ShieldCheck,
} from 'lucide-react'
import { useAuth } from '../../auth/AuthContext'

const NAV = [
  {
    label: 'Dashboard',
    to: '/dashboard',
    icon: LayoutDashboard,
    roles: ['ADMIN', 'BANK_STAFF'],
  },
  {
    label: 'Onboarding',
    to: '/onboarding',
    icon: Upload,
    roles: ['CUSTOMER'],
  },
  {
    label: 'My Verification',
    to: '/verification',
    icon: CheckCircle,
    roles: ['CUSTOMER'],
  },
  {
    label: 'Audit Logs',
    to: '/audit',
    icon: ClipboardList,
    roles: ['ADMIN'],
  },
  {
    label: 'Chatbot',
    to: '/chatbot',
    icon: MessageSquare,
    roles: ['ADMIN', 'BANK_STAFF', 'CUSTOMER'],
  },
]

export default function Sidebar() {
  const { user, logout } = useAuth()

  const visibleItems = NAV.filter((item) => item.roles.includes(user?.role))

  return (
    <aside className="w-60 shrink-0 flex flex-col min-h-screen bg-[var(--color-ink)] text-white">
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-5 border-b border-white/10">
        <ShieldCheck className="w-7 h-7 text-[var(--color-accent)]" />
        <div>
          <p className="text-xs font-bold tracking-widest uppercase text-white/50">
            BankVerify
          </p>
          <p className="text-sm font-semibold leading-tight">KYC Suite</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {visibleItems.map(({ label, to, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-white/10 text-white font-medium'
                  : 'text-white/60 hover:bg-white/5 hover:text-white'
              }`
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* User + logout */}
      <div className="px-6 py-4 border-t border-white/10">
        <p className="text-xs text-white/50 truncate mb-0.5">{user?.email}</p>
        <p className="text-xs font-semibold text-[var(--color-accent)] mb-3">
          {user?.role}
        </p>
        <button
          onClick={logout}
          className="flex items-center gap-2 text-xs text-white/50 hover:text-white transition-colors"
        >
          <LogOut className="w-3.5 h-3.5" />
          Sign out
        </button>
      </div>
    </aside>
  )
}
