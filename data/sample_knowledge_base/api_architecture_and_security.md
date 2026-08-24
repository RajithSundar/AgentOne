# API Architecture, Rate Limiting & Security Governance

## 1. Rate Limiting Limits
- **Enterprise Plan**: 10,000 requests / minute, burst capacity up to 15,000 req/min.
- **Pro Plan**: 2,500 requests / minute, burst capacity up to 3,500 req/min.
- **Developer / Starter Plan**: 600 requests / minute.
- Exceeding limits returns HTTP 429 `Too Many Requests` with a `Retry-After` header indicating backoff duration in seconds.

## 2. Authentication & Token Management
- All REST and SSE streaming endpoints require bearer token authentication: `Authorization: Bearer <API_KEY>`.
- Internal service-to-service communication uses mTLS with automated 30-day certificate rotation.

## 3. Data Protection & Privacy Guardrails
- Personally Identifiable Information (PII) including Social Security Numbers, credit card numbers, and raw phone numbers are masked at ingestion using regex and NLP entropy filters before reaching downstream reasoning models.
- All storage at rest is encrypted with AES-256; vector embeddings do not persist raw plaintext without token pseudonymization.
