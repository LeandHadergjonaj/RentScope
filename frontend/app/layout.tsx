import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'RentScope - Property Rent Evaluation',
  description: 'Evaluate rental property prices in London',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

