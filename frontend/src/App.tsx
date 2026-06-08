import { Navigate, Route, Routes } from 'react-router-dom'

import { ProtectedRoute } from '@/components/ProtectedRoute'
import { Chats } from '@/pages/Chats'
import { Login } from '@/pages/Login'
import { SignUp } from '@/pages/SignUp'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<SignUp />} />
      <Route
        path="/chats"
        element={
          <ProtectedRoute>
            <Chats />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/chats" replace />} />
    </Routes>
  )
}
