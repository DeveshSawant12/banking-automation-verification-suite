import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import {
  FileText, Shield, User, Activity, Brain, AlertTriangle, CheckCircle, XCircle,
} from 'lucide-react'
import {
  getCase, getFraudScore, getOcrResult, getTamperingResult,
  getFaceMatchResult, getLivenessResult, getCrossDocResult,
  getGradCamResult, makeCaseDecision,
} from '../api/client'
import { Card, Spinner, EmptyState } from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import RiskScoreGauge from '../components/verification/RiskScoreGauge'
import VerificationCard from '../components/verification/VerificationCard'
import { useAuth } from '../auth/AuthContext'

function Section({ title, icon: Icon, children }) {
  return (
    <div className="bg-white rounded-xl border border-[var(--color-border)] p-6">
      <div className="flex items-center gap-2 mb-5">
        <Icon className="w-4 h-4 text-[var(--color-primary)]" />
        <h2 className="text-sm font-bold text-[var(--color-ink)] uppercase tracking-wide">
          {title}
        </h2>
      </div>
      {children}
    </div>
  )
}

function DataRow({ label, value, mono }) {
  return (
    <div className="flex justify-between py-2 border-b border-[var(--color-border)] last:border-0">
      <span className="text-xs text-[var(--color-muted)]">{label}</span>
      <span className={`text-xs font-medium text-[var(--color-ink)] ${mono ? 'font-mono' : ''}`}>
        {value ?? '—'}
      </span>
    </div>
  )
}

function ScoreBreakdownTable({ breakdown }) {
  const EXCLUDED_KEYS = ['raw_total_before_clamp', 'face_verification_detail', 'cross_document_detail']
  const rows = Object.entries(breakdown).filter(([k]) => !EXCLUDED_KEYS.includes(k))

  return (
    <div className="space-y-0 divide-y divide-[var(--color-border)]">
      {rows.map(([key, val]) => {
        if (typeof val !== 'number') return null
        const label = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
        return (
          <div key={key} className="flex justify-between items-center py-2">
            <span className="text-xs text-[var(--color-muted)]">{label}</span>
            <span className={`text-xs font-bold ${val > 0 ? 'text-[var(--color-danger)]' : 'text-[var(--color-accent)]'}`}>
              {val > 0 ? `+${val}` : '✓'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export default function CaseDetail() {
  const { caseId } = useParams()
  const { hasRole } = useAuth()
  const isStaff = hasRole('ADMIN', 'BANK_STAFF')

  const [caseData, setCaseData] = useState(null)
  const [fraudScore, setFraudScore] = useState(null)
  const [ocr, setOcr] = useState(null)
  const [tampering, setTampering] = useState(null)
  const [faceMatch, setFaceMatch] = useState(null)
  const [liveness, setLiveness] = useState(null)
  const [crossDoc, setCrossDoc] = useState(null)
  const [gradcam, setGradcam] = useState(null)
  const [loading, setLoading] = useState(true)
  const [deciding, setDeciding] = useState(false)
  const [decisionError, setDecisionError] = useState(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      const settled = await Promise.allSettled([
        getCase(caseId),
        getFraudScore(caseId),
        getOcrResult(caseId),
        getTamperingResult(caseId),
        getFaceMatchResult(caseId),
        getLivenessResult(caseId),
        getCrossDocResult(caseId),
        getGradCamResult(caseId),
      ])
      if (settled[0].status === 'fulfilled') setCaseData(settled[0].value.data)
      if (settled[1].status === 'fulfilled') setFraudScore(settled[1].value.data)
      if (settled[2].status === 'fulfilled') setOcr(settled[2].value.data)
      if (settled[3].status === 'fulfilled') setTampering(settled[3].value.data)
      if (settled[4].status === 'fulfilled') setFaceMatch(settled[4].value.data)
      if (settled[5].status === 'fulfilled') setLiveness(settled[5].value.data)
      if (settled[6].status === 'fulfilled') setCrossDoc(settled[6].value.data)
      if (settled[7].status === 'fulfilled') setGradcam(settled[7].value.data)
      setLoading(false)
    }
    load()
  }, [caseId])

  async function handleDecision(decision) {
    setDecisionError(null)
    setDeciding(true)
    try {
      const res = await makeCaseDecision(caseId, decision)
      setCaseData(res.data)
    } catch (err) {
      setDecisionError(err.response?.data?.detail || 'Decision failed. Please try again.')
    } finally {
      setDeciding(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner />
      </div>
    )
  }

  if (!caseData) {
    return (
      <EmptyState
        icon={AlertTriangle}
        title="Case not found"
        description="This case may have been deleted or you may not have permission to view it."
      />
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-ink)]">
            Case <span className="font-mono text-base">{caseData.id.slice(0, 8)}…</span>
          </h1>
          <div className="flex items-center gap-3 mt-1.5">
            <Badge label={caseData.status} />
            <span className="text-xs text-[var(--color-muted)]">
              Submitted {new Date(caseData.created_at).toLocaleString('en-IN')}
            </span>
          </div>
        </div>

        {/* Staff decision controls */}
        {isStaff && ['REVIEW_REQUIRED', 'PENDING', 'PROCESSING'].includes(caseData.status) && (
          <div className="flex gap-2">
            <button
              onClick={() => handleDecision('APPROVED')}
              disabled={deciding}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[var(--color-accent)]
                         hover:opacity-90 disabled:opacity-60 text-white text-xs font-semibold
                         transition-opacity"
            >
              <CheckCircle className="w-3.5 h-3.5" />
              Approve
            </button>
            <button
              onClick={() => handleDecision('REJECTED')}
              disabled={deciding}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[var(--color-danger)]
                         hover:opacity-90 disabled:opacity-60 text-white text-xs font-semibold
                         transition-opacity"
            >
              <XCircle className="w-3.5 h-3.5" />
              Reject
            </button>
          </div>
        )}
      </div>

      {decisionError && (
        <div className="px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
          {decisionError}
        </div>
      )}

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fraud Score */}
        {fraudScore && (
          <Section title="Fraud Risk Score" icon={Shield}>
            <RiskScoreGauge score={fraudScore.total_score} riskLevel={fraudScore.risk_level} />
            <div className="mt-5">
              <p className="text-xs font-semibold text-[var(--color-muted)] uppercase tracking-wide mb-2">
                Score Breakdown
              </p>
              <ScoreBreakdownTable breakdown={fraudScore.score_breakdown} />
            </div>
            {fraudScore.requires_manual_review && (
              <div className="mt-4 px-3 py-2.5 rounded-lg bg-amber-50 border border-amber-200 flex items-center gap-2">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                <span className="text-xs font-medium text-amber-700">
                  Manual review required
                </span>
              </div>
            )}
          </Section>
        )}

        {/* OCR Results */}
        {ocr && (
          <Section title="Extracted Fields" icon={FileText}>
            <DataRow label="Name" value={ocr.extracted_name} />
            <DataRow label="Date of Birth" value={ocr.extracted_dob} />
            <DataRow label="Gender" value={ocr.extracted_gender} />
            <DataRow label="Aadhaar No." value={ocr.extracted_aadhaar_no} mono />
            <DataRow label="PAN No." value={ocr.extracted_pan_no} mono />
            <DataRow label="Address" value={ocr.extracted_address} />
          </Section>
        )}

        {/* Verification Summary */}
        <Section title="Verification Checks" icon={Activity}>
          <div className="space-y-3">
            {tampering && (
              <VerificationCard
                title="Aadhaar Tampering"
                verdict={tampering.aadhaar?.verdict}
                details={[
                  { label: 'Confidence', value: `${((tampering.aadhaar?.confidence || 0) * 100).toFixed(1)}%` },
                  { label: 'ELA Score', value: tampering.aadhaar?.ela_score?.toFixed(2) ?? '—' },
                ]}
              />
            )}
            {tampering && (
              <VerificationCard
                title="PAN Tampering"
                verdict={tampering.pan?.verdict}
                details={[
                  { label: 'Confidence', value: `${((tampering.pan?.confidence || 0) * 100).toFixed(1)}%` },
                  { label: 'ELA Score', value: tampering.pan?.ela_score?.toFixed(2) ?? '—' },
                ]}
              />
            )}
            {faceMatch && (
              <VerificationCard
                title="Face Match"
                verdict={faceMatch.overall_match ? 'MATCH' : 'MISMATCH'}
                details={[
                  { label: 'Avg. Similarity', value: `${faceMatch.average_match_pct?.toFixed(1) ?? '—'}%` },
                  { label: 'Comparisons', value: faceMatch.comparisons_run ?? '—' },
                ]}
              />
            )}
            {liveness && (
              <VerificationCard
                title="Liveness"
                verdict={liveness.verdict}
                details={[
                  { label: 'Blink', value: liveness.blink_detected ? 'Detected' : 'Not detected' },
                  { label: 'Head movement', value: liveness.head_movement_detected ? 'Detected' : 'Not detected' },
                  { label: 'Smile', value: liveness.smile_detected ? 'Detected' : 'Not detected' },
                ]}
              />
            )}
            {crossDoc && (
              <VerificationCard
                title="Cross-Document Match"
                verdict={crossDoc.name_match_score >= 75 && crossDoc.dob_match ? 'MATCH' : 'MISMATCH'}
                details={[
                  { label: 'Name score', value: `${crossDoc.name_match_score?.toFixed(0) ?? '—'}/100` },
                  { label: 'DOB match', value: crossDoc.dob_match ? 'Yes' : 'No' },
                ]}
              />
            )}
          </div>
        </Section>
      </div>

      {/* Grad-CAM Visualization */}
      {gradcam && (
        <Section title="Explainability — Grad-CAM Heatmap" icon={Brain}>
          <div className="space-y-3">
            <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 px-3 py-2 rounded-lg">
              {gradcam.disclaimer}
            </p>
            {gradcam.gradcam_heatmap_url && (
              <img
                src={gradcam.gradcam_heatmap_url}
                alt="Grad-CAM heatmap overlay showing regions of interest"
                className="rounded-lg border border-[var(--color-border)] max-w-full"
              />
            )}
            <div className="flex gap-6 text-xs text-[var(--color-muted)]">
              <span>ImageNet class predicted: #{gradcam.predicted_imagenet_class_idx}</span>
              <span>Confidence: {(gradcam.predicted_imagenet_class_confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </Section>
      )}

      {/* ELA Feature Explanation (primary explanation) */}
      {fraudScore?.score_breakdown && (
        <Section title="Primary Explanation — Feature Attribution" icon={Brain}>
          <p className="text-xs text-[var(--color-muted)] mb-3">
            This explanation is based on the Random Forest classifier's feature importances
            and the actual feature values computed for this document. Unlike Grad-CAM
            above, this directly explains why the tampering verdict was produced.
          </p>
          <div className="grid grid-cols-3 gap-4 text-center">
            {['ela', 'resnet', 'ocr'].map((group) => {
              const prefix = group === 'resnet' ? 'resnet_feat_' : `${group}_`
              const groupScore = Object.entries(fraudScore.score_breakdown)
                .filter(([k]) => k.startsWith(prefix))
                .reduce((s, [, v]) => s + (typeof v === 'number' ? v : 0), 0)
              return (
                <div key={group} className="bg-[var(--color-surface)] rounded-lg p-3">
                  <p className="text-xs font-semibold text-[var(--color-ink)] uppercase tracking-wide">
                    {group === 'ela' ? 'ELA' : group === 'resnet' ? 'ResNet18' : 'OCR'} Group
                  </p>
                  <p className="text-lg font-bold text-[var(--color-primary)] mt-1">
                    {groupScore.toFixed(4)}
                  </p>
                  <p className="text-xs text-[var(--color-muted)]">combined importance</p>
                </div>
              )
            })}
          </div>
        </Section>
      )}
    </div>
  )
}
