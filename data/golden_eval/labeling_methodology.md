# Golden Evaluation Set — Labeling Methodology

## Sampling strategy

- **Source**: 5,000 cleaned AppleSupport conversation pairs from Twitter
- **Strategy**: Stratified sampling ~20 per intent (10 intents), plus 20 additional 
  escalation-positive edge cases to ensure adequate representation
- **Total**: 220 labeled examples
- **Seed**: 42 for reproducibility

## Labeling process

1. **Initial classification**: keyword-based classifier assigns an auto-intent
2. **Manual review pass**: each example's intent was reviewed against the taxonomy,
   with corrections applied for:
   - Multi-signal messages (e.g. "battery drain after update" → battery_power, not software_update)
   - Domain-specific terms (e.g. "Apple Music" → audio_media, not app_issue)
   - Ambiguous cases resolved by the PRIMARY complaint
3. **Escalation labeling**: based on content analysis following the escalation rules:
   - Hardware defects requiring physical inspection → escalate
   - Account security concerns (locked, hacked) → escalate
   - Financial disputes (refund, charges) → escalate
   - Extreme frustration expressed → escalate
   - Repeated failed troubleshooting → escalate
   - Data loss/privacy concerns → escalate
   - Standard troubleshooting → auto-handle

## Honest limitations

- Intent labels were produced by a careful rule-based system with manual corrections,
  not by a team of independent human annotators
- Inter-annotator agreement was not measured (single annotator)
- Some borderline cases (e.g. "my phone is slow after the update" — performance or 
  software_update?) are genuinely ambiguous; we defaulted to the primary symptom
- The escalation threshold is subjective; we erred on the side of escalation for 
  safety-critical cases (account, hardware, billing)

## Distribution

| Intent | Count |
|--------|-------|
| general_inquiry | 26 |
| audio_media | 24 |
| software_update | 24 |
| battery_power | 23 |
| hardware | 23 |
| account_icloud | 22 |
| performance | 20 |
| screen_display | 20 |
| app_issue | 19 |
| connectivity | 19 |

Escalation: 58/220 (26%) flagged for human handling
