import { useState, useEffect, useRef, useCallback } from 'react'
import { Send, MessageSquare, Bot, User, ChevronDown, BookOpen } from 'lucide-react'
import { createChatSession, sendChatMessage, getChatHistory } from '../api/client'
import { Spinner } from '../components/ui/Card'

function Message({ msg }) {
  const isUser = msg.role === 'user'
  const [showSources, setShowSources] = useState(false)
  const chunks = msg.retrieved_chunks_json?.chunks || []

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      <div className={`w-7 h-7 rounded-full shrink-0 flex items-center justify-center ${
        isUser ? 'bg-[var(--color-primary)]' : 'bg-[var(--color-ink)]'
      }`}>
        {isUser
          ? <User className="w-3.5 h-3.5 text-white" />
          : <Bot className="w-3.5 h-3.5 text-white" />}
      </div>

      <div className={`max-w-[75%] space-y-2 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
          isUser
            ? 'bg-[var(--color-primary)] text-white rounded-tr-sm'
            : 'bg-white border border-[var(--color-border)] text-[var(--color-ink)] rounded-tl-sm'
        }`}>
          {msg.content}
        </div>

        {/* Source attribution */}
        {!isUser && chunks.length > 0 && (
          <div className="w-full">
            <button
              onClick={() => setShowSources((v) => !v)}
              className="flex items-center gap-1.5 text-xs text-[var(--color-muted)] hover:text-[var(--color-ink)] transition-colors"
            >
              <BookOpen className="w-3 h-3" />
              {chunks.length} source{chunks.length > 1 ? 's' : ''} retrieved
              <ChevronDown className={`w-3 h-3 transition-transform ${showSources ? 'rotate-180' : ''}`} />
            </button>

            {showSources && (
              <div className="mt-2 space-y-1.5">
                {chunks.map((chunk, idx) => (
                  <div
                    key={idx}
                    className="text-xs bg-[var(--color-surface)] border border-[var(--color-border)]
                               rounded-lg px-3 py-2"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-[var(--color-ink)]">
                        {chunk.source}
                      </span>
                      <span className="text-[var(--color-muted)]">
                        Page {chunk.page}
                      </span>
                    </div>
                    <p className="text-[var(--color-muted)] leading-relaxed line-clamp-3">
                      {chunk.content}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!isUser && msg.model_used && (
          <span className="text-xs text-[var(--color-muted)]/60">
            via {msg.model_used}
          </span>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="w-7 h-7 rounded-full bg-[var(--color-ink)] flex items-center justify-center shrink-0">
        <Bot className="w-3.5 h-3.5 text-white" />
      </div>
      <div className="bg-white border border-[var(--color-border)] rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex gap-1 items-center h-4">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-[var(--color-muted)] animate-bounce"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

const SUGGESTED_QUESTIONS = [
  'What documents are required for KYC?',
  'What are the RBI guidelines for customer due diligence?',
  'How does the video KYC process work?',
  'What is the validity period of KYC documents?',
]

export default function ChatbotPage() {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [initialising, setInitialising] = useState(true)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  // Create a session on mount
  useEffect(() => {
    async function init() {
      try {
        const res = await createChatSession()
        setSessionId(res.data.id)
        // Load any existing history
        const histRes = await getChatHistory(res.data.id)
        setMessages(histRes.data || [])
      } catch (err) {
        setError('Could not start a chat session. Please refresh and try again.')
      } finally {
        setInitialising(false)
      }
    }
    init()
  }, [])

  async function send(text) {
    const content = text.trim()
    if (!content || !sessionId || sending) return
    setInput('')
    setSending(true)
    setError(null)

    // Optimistically append user message
    const tempUserMsg = { id: `tmp-user-${Date.now()}`, role: 'user', content, created_at: new Date().toISOString() }
    setMessages((m) => [...m, tempUserMsg])

    try {
      const res = await sendChatMessage(sessionId, content)
      const { user_message, assistant_message } = res.data
      // Replace temp with real persisted messages
      setMessages((m) => [
        ...m.filter((msg) => msg.id !== tempUserMsg.id),
        user_message,
        assistant_message,
      ])
    } catch (err) {
      setMessages((m) => m.filter((msg) => msg.id !== tempUserMsg.id))
      setError(err.response?.data?.detail || 'Failed to send message. Please try again.')
    } finally {
      setSending(false)
      inputRef.current?.focus()
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="shrink-0 flex items-center gap-3 px-6 py-4 border-b border-[var(--color-border)] bg-white">
        <div className="w-9 h-9 rounded-xl bg-[var(--color-ink)] flex items-center justify-center">
          <MessageSquare className="w-4.5 h-4.5 text-white" />
        </div>
        <div>
          <p className="text-sm font-bold text-[var(--color-ink)]">Banking Assistant</p>
          <p className="text-xs text-[var(--color-muted)]">
            Ask about KYC rules, RBI guidelines, loan policies, and FAQs
          </p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5 bg-[var(--color-surface)]">
        {initialising ? (
          <div className="flex justify-center py-10"><Spinner /></div>
        ) : messages.length === 0 ? (
          <div className="text-center mt-8">
            <Bot className="w-12 h-12 text-[var(--color-muted)] mx-auto mb-3" />
            <p className="text-sm font-semibold text-[var(--color-ink)] mb-1">
              How can I help you today?
            </p>
            <p className="text-xs text-[var(--color-muted)] mb-6">
              I can answer questions about KYC procedures and banking regulations
              using the bank's policy documents.
            </p>
            <div className="grid grid-cols-2 gap-2 max-w-md mx-auto">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-left text-xs px-3 py-2.5 rounded-lg border border-[var(--color-border)]
                             bg-white hover:border-[var(--color-primary)] hover:text-[var(--color-primary)]
                             transition-colors text-[var(--color-muted)]"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => <Message key={msg.id} msg={msg} />)
        )}

        {sending && <TypingIndicator />}

        {error && (
          <div className="text-center">
            <p className="text-xs text-[var(--color-danger)] bg-red-50 border border-red-200
                          rounded-lg px-4 py-2.5 inline-block">
              {error}
            </p>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="shrink-0 px-6 py-4 border-t border-[var(--color-border)] bg-white">
        <div className="flex gap-3 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about KYC requirements, RBI guidelines, loan policies…"
            rows={1}
            className="flex-1 resize-none rounded-xl border border-[var(--color-border)] px-4 py-3
                       text-sm outline-none focus:border-[var(--color-primary)]
                       focus:ring-2 focus:ring-[var(--color-primary)]/20 transition-colors
                       min-h-[44px] max-h-32"
            style={{ height: 'auto' }}
            disabled={!sessionId || initialising}
          />
          <button
            onClick={() => send(input)}
            disabled={!input.trim() || !sessionId || sending}
            className="w-11 h-11 rounded-xl bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)]
                       disabled:opacity-40 disabled:cursor-not-allowed
                       flex items-center justify-center transition-colors shrink-0"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
        <p className="text-xs text-[var(--color-muted)] mt-2">
          Press Enter to send · Shift+Enter for a new line
        </p>
      </div>
    </div>
  )
}
