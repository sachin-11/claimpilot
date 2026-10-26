#!/usr/bin/env bash
# Writes ../.env from .env.production.example with fresh random secrets.
#
#   OPENAI_API_KEY=sk-... ./deploy/bootstrap-env.sh claims.example.com
#
# Refuses to overwrite an existing .env. Prints the seed passwords once: save them.
set -euo pipefail

domain="${1:-}"
if [[ -z "$domain" ]]; then
  echo "usage: OPENAI_API_KEY=sk-... $0 <public host name, e.g. claimpilot.duckdns.org>" >&2
  exit 1
fi
here="$(cd "$(dirname "$0")" && pwd)"
target="$here/../.env"
if [[ -e "$target" ]]; then
  echo "$target already exists; move it away first" >&2
  exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -r -s -p "OpenAI API key: " OPENAI_API_KEY
  echo
fi

rand() { openssl rand -hex "$1"; }
admin_password="$(rand 12)"
officer_password="$(rand 12)"

sed \
  -e "s|__DOMAIN__|$domain|g" \
  -e "s|__POSTGRES_PASSWORD__|$(rand 16)|g" \
  -e "s|__JWT_SECRET__|$(rand 32)|g" \
  -e "s|__ADMIN_PASSWORD__|$admin_password|g" \
  -e "s|__OFFICER_PASSWORD__|$officer_password|g" \
  -e "s|__OPENAI_API_KEY__|$OPENAI_API_KEY|g" \
  -e "s|__TOKEN_COVERAGE__|$(rand 24)|g" \
  -e "s|__TOKEN_INTAKE__|$(rand 24)|g" \
  -e "s|__TOKEN_RISK__|$(rand 24)|g" \
  -e "s|__TOKEN_DECISION__|$(rand 24)|g" \
  -e "s|__TOKEN_SUPERVISOR__|$(rand 24)|g" \
  -e "s|__TOKEN_NOTIFY__|$(rand 24)|g" \
  -e "s|__METRICS_TOKEN__|$(rand 24)|g" \
  "$here/.env.production.example" > "$target"
chmod 600 "$target"

if grep -q "__[A-Z_]*__" "$target"; then
  echo "unfilled markers left in $target:" >&2
  grep -n "__[A-Z_]*__" "$target" >&2
  exit 1
fi

echo "wrote $target"
echo
echo "Seed logins (save them, they are not shown again):"
echo "  admin@example.com    $admin_password"
echo "  officer@example.com  $officer_password"
