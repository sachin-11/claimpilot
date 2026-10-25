// Regenerates src/api/schema.d.ts from the running backend (or OPENAPI_URL / a local openapi.json).
import { spawnSync } from 'node:child_process'

const source = process.env.OPENAPI_URL ?? 'http://localhost:8010/openapi.json'
const bin = process.platform === 'win32' ? 'openapi-typescript.cmd' : 'openapi-typescript'
const result = spawnSync(`npx ${bin} "${source}" -o src/api/schema.d.ts`, {
  stdio: 'inherit',
  shell: true,
})
process.exit(result.status ?? 1)
