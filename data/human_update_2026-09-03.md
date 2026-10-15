# Human-data update audit: 2026-09-03

Source: `data_backup_2026-09-03`

The manuscript-facing audit identified 109 completed sessions and 218 completed case trajectories. Inclusion followed the existing rule `completion_status == "completed"`; recommendation analyses used actual delivered case-level exposure rather than nominal assignment.

| Quantity | V2 (2026-08-25) | V3 (2026-09-03) |
|---|---:|---:|
| Completed participants | 50 | 109 |
| Completed case trajectories | 100 | 218 |
| State ratings | 423 | 941 |
| Baseline ratings | 78 | 184 |
| Post-turn ratings | 345 | 757 |
| Trajectories with post-turn reports | 69 | 159 |
| Recommendation-exposed trajectories | 36 | 82 |
| Viral-gastroenteritis factorial trajectories | 21 | 46 |

The updated viral-gastroenteritis comparison changes the qualitative description of the AI-confidence--information-need association. The overall high-minus-low contrast remains negative, but the direction differs by recommendation level: it is negative under lower recommendations and slightly positive under higher recommendations. The manuscript therefore describes this association as condition-dependent and does not treat it as demonstrated transfer.

This audit contains aggregate quantities only. It excludes participant identifiers, session identifiers, message text, timestamps, and raw event records.
