import { Navigate, Route, Routes } from 'react-router-dom'

import { ProtectedRoute } from '@/components/ProtectedRoute'
import { Login } from '@/pages/Login'
import { SignUp } from '@/pages/SignUp'
import { ChatLayout } from '@/components/chat/ChatLayout'
import { ChatEmptyPage } from '@/pages/chat/ChatEmptyPage'
import { ChatThreadPage } from '@/pages/chat/ChatThreadPage'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<SignUp />} />
      <Route
        path="/chats"
        element={
          <ProtectedRoute>
            <ChatLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<ChatEmptyPage />} />
        <Route path=":threadId" element={<ChatThreadPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/chats" replace />} />
    </Routes>
  )
}
