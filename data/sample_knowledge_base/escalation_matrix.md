# Customer Support Escalation & Routing Matrix

## 1. Routing Tiers
- **Tier 1 - Automated Multi-Agent Resolution**: Handles common inquiries, password resets, plan documentation, and standard knowledge queries with zero latency.
- **Tier 2 - Senior Customer Operations**: Assigned when user sentiment drops below -0.60 or when complex multi-step account reconciliation is required.
- **Tier 3 - Engineering & Infrastructure On-Call**: Assigned for system bugs, 5xx outages, database replication alerts, and webhook failures.

## 2. High-Impact Action Governance
The following operations MUST trigger approval gates:
1. `refund_process` where amount >= $500.00 USD
2. `delete_user` or `wipe_workspace_data`
3. `update_rbac` to admin/owner roles
4. `database_schema_migration` or live cache flush
