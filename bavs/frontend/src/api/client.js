import axios from 'axios'

const client = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
})

// Attach stored access token to every request
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// On 401, clear auth state and redirect to login
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Auth
export const login = (email, password) =>
  client.post('/auth/login', { email, password })

export const getMe = () => client.get('/auth/me')

// KYC Cases
export const createCase = () => client.post('/kyc/cases')
export const uploadDocument = (caseId, file, documentType) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('document_type', documentType)
  return client.post(`/kyc/cases/${caseId}/documents`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
export const runPipeline = (caseId) =>
  client.post(`/kyc/cases/${caseId}/run-pipeline`)
export const getCaseStatus = (caseId) =>
  client.get(`/kyc/cases/${caseId}/status`)
export const getCase = (caseId) => client.get(`/kyc/cases/${caseId}`)
export const listCases = (params) => client.get('/kyc/cases', { params })
export const makeCaseDecision = (caseId, decision) =>
  client.patch(`/kyc/cases/${caseId}/decision`, { decision })

// Verification results
export const getOcrResult = (caseId) =>
  client.get(`/verification/${caseId}/ocr`)
export const getTamperingResult = (caseId) =>
  client.get(`/verification/${caseId}/tampering`)
export const getFaceMatchResult = (caseId) =>
  client.get(`/verification/${caseId}/face-match`)
export const getCrossDocResult = (caseId) =>
  client.get(`/verification/${caseId}/cross-document`)
export const getLivenessResult = (caseId) =>
  client.get(`/verification/${caseId}/liveness`)
export const getGradCamResult = (caseId) =>
  client.get(`/verification/${caseId}/gradcam`)

// Fraud
export const getFraudScore = (caseId) =>
  client.get(`/fraud/${caseId}/score`)

// Audit logs
export const getAuditLogs = (params) => client.get('/audit/logs', { params })
export const getCaseAuditLogs = (caseId) =>
  client.get(`/audit/logs/${caseId}`)

// Admin stats
export const getAdminSummary = () => client.get('/admin/stats/summary')
export const getRiskDistribution = () =>
  client.get('/admin/stats/risk-distribution')

// Chatbot
export const createChatSession = () => client.post('/chatbot/sessions')
export const sendChatMessage = (sessionId, content) =>
  client.post(`/chatbot/sessions/${sessionId}/messages`, { content })
export const getChatHistory = (sessionId) =>
  client.get(`/chatbot/sessions/${sessionId}/messages`)

export default client
