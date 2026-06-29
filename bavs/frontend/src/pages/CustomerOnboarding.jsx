import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, CheckCircle, Loader, FileText, Camera } from 'lucide-react'
import {
  createCase,
  uploadDocument,
  runPipeline,
  getCaseStatus,
} from '../api/client'

const STEPS = [
  { key: 'AADHAAR', label: 'Aadhaar Card', icon: FileText, hint: 'Upload front side of your Aadhaar card' },
  { key: 'PAN', label: 'PAN Card', icon: FileText, hint: 'Upload your PAN card' },
  { key: 'SELFIE', label: 'Selfie', icon: Camera, hint: 'Upload a clear, well-lit selfie' },
]

function StepIndicator({ current, steps }) {
  return (
    <div className="flex items-center gap-0 mb-8">
      {steps.map((step, idx) => (
        <div key={step.key} className="flex items-center flex-1">
          <div className={`flex flex-col items-center ${idx < steps.length - 1 ? 'flex-1' : ''}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-colors ${
              idx < current
                ? 'bg-[var(--color-accent)] text-white'
                : idx === current
                ? 'bg-[var(--color-primary)] text-white'
                : 'bg-[var(--color-border)] text-[var(--color-muted)]'
            }`}>
              {idx < current ? <CheckCircle className="w-4 h-4" /> : idx + 1}
            </div>
            <span className={`mt-1.5 text-xs text-center leading-tight ${
              idx === current ? 'text-[var(--color-ink)] font-semibold' : 'text-[var(--color-muted)]'
            }`}>
              {step.label}
            </span>
          </div>
          {idx < steps.length - 1 && (
            <div className={`h-0.5 flex-1 mx-2 mt-[-18px] transition-colors ${
              idx < current ? 'bg-[var(--color-accent)]' : 'bg-[var(--color-border)]'
            }`} />
          )}
        </div>
      ))}
    </div>
  )
}

function DropZone({ onFile, file, accept }) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) onFile(dropped)
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
      className={`cursor-pointer border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
        dragging
          ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/5'
          : file
          ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/5'
          : 'border-[var(--color-border)] hover:border-[var(--color-primary)] hover:bg-[var(--color-surface)]'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
      {file ? (
        <div className="flex flex-col items-center gap-2">
          <CheckCircle className="w-10 h-10 text-[var(--color-accent)]" />
          <p className="text-sm font-medium text-[var(--color-ink)]">{file.name}</p>
          <p className="text-xs text-[var(--color-muted)]">
            {(file.size / 1024).toFixed(0)} KB · Click to change
          </p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2">
          <Upload className="w-10 h-10 text-[var(--color-muted)]" />
          <p className="text-sm font-medium text-[var(--color-ink)]">
            Drop file here or click to browse
          </p>
          <p className="text-xs text-[var(--color-muted)]">JPG, PNG up to 10 MB</p>
        </div>
      )}
    </div>
  )
}

const POLL_INTERVAL_MS = 3000
const MAX_POLLS = 40  // give up after ~2 min

export default function CustomerOnboarding() {
  const navigate = useNavigate()
  const [step, setStep] = useState(0)
  const [files, setFiles] = useState({ AADHAAR: null, PAN: null, SELFIE: null })
  const [caseId, setCaseId] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [polling, setPolling] = useState(false)
  const [caseStatus, setCaseStatus] = useState(null)
  const [error, setError] = useState(null)

  function setFile(docType, file) {
    setFiles((f) => ({ ...f, [docType]: file }))
  }

  async function handleNext() {
    const docType = STEPS[step].key
    if (!files[docType]) {
      setError('Please select a file before continuing.')
      return
    }
    setError(null)
    setSubmitting(true)

    try {
      // Create case on first step
      let activeCaseId = caseId
      if (!activeCaseId) {
        const res = await createCase()
        activeCaseId = res.data.id
        setCaseId(activeCaseId)
      }

      await uploadDocument(activeCaseId, files[docType], docType)

      if (step < STEPS.length - 1) {
        setStep((s) => s + 1)
      } else {
        // All files uploaded — trigger pipeline and start polling
        await runPipeline(activeCaseId)
        setPolling(true)
        pollStatus(activeCaseId)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  function pollStatus(id, count = 0) {
    if (count >= MAX_POLLS) {
      setPolling(false)
      setError('Verification is taking longer than expected. Check back later.')
      return
    }
    setTimeout(async () => {
      try {
        const res = await getCaseStatus(id)
        const status = res.data.status
        setCaseStatus(status)
        if (status === 'PENDING' || status === 'PROCESSING') {
          pollStatus(id, count + 1)
        } else {
          setPolling(false)
          navigate(`/cases/${id}`)
        }
      } catch {
        pollStatus(id, count + 1)
      }
    }, POLL_INTERVAL_MS)
  }

  if (polling) {
    return (
      <div className="max-w-md mx-auto mt-20 text-center">
        <Loader className="w-10 h-10 text-[var(--color-primary)] animate-spin mx-auto mb-4" />
        <h2 className="text-lg font-bold text-[var(--color-ink)]">
          Verifying your documents…
        </h2>
        <p className="text-sm text-[var(--color-muted)] mt-1">
          {caseStatus === 'PROCESSING'
            ? 'AI pipeline is running. This takes 30–60 seconds.'
            : 'Pipeline queued. Starting shortly…'}
        </p>
        <p className="text-xs text-[var(--color-muted)] mt-4">
          You can leave this page — the pipeline continues in the background.
        </p>
      </div>
    )
  }

  const currentStep = STEPS[step]
  const StepIcon = currentStep.icon

  return (
    <div className="max-w-lg mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-[var(--color-ink)]">Complete Your KYC</h1>
        <p className="text-sm text-[var(--color-muted)] mt-0.5">
          Upload your documents to begin the verification process.
        </p>
      </div>

      <div className="bg-white rounded-2xl border border-[var(--color-border)] shadow-sm p-8">
        <StepIndicator current={step} steps={STEPS} />

        {error && (
          <div className="mb-5 px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="flex items-center gap-3 mb-5">
          <div className="w-9 h-9 rounded-lg bg-[var(--color-surface)] flex items-center justify-center">
            <StepIcon className="w-4.5 h-4.5 text-[var(--color-primary)]" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[var(--color-ink)]">
              {currentStep.label}
            </p>
            <p className="text-xs text-[var(--color-muted)]">{currentStep.hint}</p>
          </div>
        </div>

        <DropZone
          onFile={(f) => setFile(currentStep.key, f)}
          file={files[currentStep.key]}
          accept="image/jpeg,image/png,image/webp"
        />

        <div className="flex gap-3 mt-6">
          {step > 0 && (
            <button
              onClick={() => { setStep((s) => s - 1); setError(null) }}
              className="flex-1 py-2.5 rounded-lg border border-[var(--color-border)] text-sm
                         font-semibold text-[var(--color-ink)] hover:bg-[var(--color-surface)] transition-colors"
            >
              Back
            </button>
          )}
          <button
            onClick={handleNext}
            disabled={submitting || !files[currentStep.key]}
            className="flex-1 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)]
                       disabled:opacity-60 disabled:cursor-not-allowed
                       text-white text-sm font-semibold py-2.5 rounded-lg transition-colors
                       flex items-center justify-center gap-2"
          >
            {submitting && <Loader className="w-4 h-4 animate-spin" />}
            {step < STEPS.length - 1 ? 'Next' : 'Submit & Verify'}
          </button>
        </div>
      </div>
    </div>
  )
}
