import { useEffect, useState, type ReactNode } from 'react'
import { Navigate } from 'react-router-dom'

import { supabase } from '@/lib/supabase'

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<'loading' | 'authenticated' | 'unauthenticated'>('loading')

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setStatus(data.session ? 'authenticated' : 'unauthenticated')
    })

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setStatus(session ? 'authenticated' : 'unauthenticated')
    })

    return () => listener.subscription.unsubscribe()
  }, [])

  if (status === 'loading') return null
  if (status === 'unauthenticated') return <Navigate to="/login" replace />
  return <>{children}</>
}
