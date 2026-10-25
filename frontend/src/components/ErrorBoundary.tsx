import { Component, type ErrorInfo, type ReactNode } from 'react'

import { Button, Card } from '@/components/ui'

interface Props {
  children: ReactNode
  /** what failed, for the message ("the document viewer") */
  what?: string
  /** a changing key resets the boundary (e.g. the route) */
  resetKey?: unknown
}

/** A rendering error shows a message with a retry instead of a blank page. */
export class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('render error', error, info.componentStack)
  }

  componentDidUpdate(previous: Props) {
    if (this.state.error && previous.resetKey !== this.props.resetKey) this.setState({ error: null })
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <Card role="alert" className="p-6 text-center">
        <p className="font-medium">Something went wrong in {this.props.what ?? 'this part of the page'}.</p>
        <p className="mx-auto mt-1 max-w-md text-muted">{this.state.error.message}</p>
        <Button className="mt-3" size="sm" onClick={() => this.setState({ error: null })}>
          Try again
        </Button>
      </Card>
    )
  }
}
