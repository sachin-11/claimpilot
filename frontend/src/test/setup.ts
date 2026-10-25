import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

afterEach(() => {
  cleanup()
})

// jsdom has no ResizeObserver / scrollIntoView
class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= RO as unknown as typeof ResizeObserver
Element.prototype.scrollIntoView ??= () => {}

// Node's Request (what jsdom runs on) refuses relative URLs; a browser resolves them against the page.
const NativeRequest = globalThis.Request
class BrowserLikeRequest extends NativeRequest {
  constructor(input: RequestInfo | URL, init?: RequestInit) {
    super(typeof input === 'string' && input.startsWith('/') ? new URL(input, window.location.origin).toString() : input, init)
  }
}
globalThis.Request = BrowserLikeRequest as typeof Request
