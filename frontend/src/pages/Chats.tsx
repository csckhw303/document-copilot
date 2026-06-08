import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { supabase } from '@/lib/supabase'

type MeResponse = { id: string; email: string }

export function Chats() {
  const [me, setMe] = useState<MeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.get<MeResponse>('/me')
      .then(setMe)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Unknown error'))
  }, [])

  async function handleSignOut() {
    await supabase.auth.signOut()
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-8">
      <div className="w-full max-w-sm space-y-4 text-center">
        <h1 className="text-2xl font-semibold">Signed in</h1>

        {me ? (
          <p className="text-sm text-muted-foreground">
            Backend confirmed: <span className="font-medium text-foreground">{me.email}</span>
          </p>
        ) : error ? (
          <p className="text-sm text-destructive" role="alert">
            Backend error: {error}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">Verifying token…</p>
        )}

        <Button variant="outline" className="w-full" onClick={handleSignOut}>
          Sign out
        </Button>
      </div>
    </div>
  )
}
