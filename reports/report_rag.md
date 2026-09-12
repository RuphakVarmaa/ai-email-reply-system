# Evaluation Report — mode: `rag`

**Test set size**: 19  
**Mean composite**: 0.9135 (95% CI: [0.8560, 0.9439])  
**Gate failure rate**: 5.26%  
**Elapsed**: ?s

## Mean Components

| Metric | Weight | Mean |
|--------|--------|------|
| coverage | 0.30 | 1.0000 |
| faithfulness | 0.25 | 0.9500 |
| action_correctness | 0.20 | 1.0000 |
| tone | 0.15 | 0.8000 |
| entity_accuracy | 0.10 | 0.7983 |

## By Intent

| Intent | Mean Composite |
|--------|---------------|
| defect_warranty | 0.8367 |
| refund_request | 0.9575 |
| shipping_delay | 0.9242 |

## Score Distribution

Histogram [0, .2, .4, .6, .8, 1.0]: [0, 0, 1, 0, 18]  
Below 0.50: 1  
Above 0.80: 18

## Per-Response Scores (top 5 + bottom 5)

### Bottom 5

- **defect_warranty-88249770** (defect_warranty, terse): composite=0.4537  cov=1.00 faith=0.95 act=1.00 tone=0.80
  ⚠️ Gates: 1 major fact-check failure(s): reply claims 18-month warranty; KB says 24
- **shipping_delay-49225367** (shipping_delay, professional): composite=0.9242  cov=1.00 faith=0.95 act=1.00 tone=0.80
- **shipping_delay-88635248** (shipping_delay, chatty): composite=0.9242  cov=1.00 faith=0.95 act=1.00 tone=0.80
- **shipping_delay-52083301** (shipping_delay, professional): composite=0.9242  cov=1.00 faith=0.95 act=1.00 tone=0.80
- **shipping_delay-99820140** (shipping_delay, furious): composite=0.9242  cov=1.00 faith=0.95 act=1.00 tone=0.80

### Top 5

- **refund_request-47320424** (refund_request, frustrated): composite=0.9575  cov=1.00 faith=0.95 act=1.00 tone=0.80
- **refund_request-41628313** (refund_request, terse): composite=0.9575  cov=1.00 faith=0.95 act=1.00 tone=0.80
- **refund_request-25934724** (refund_request, calm): composite=0.9575  cov=1.00 faith=0.95 act=1.00 tone=0.80
- **refund_request-12408191** (refund_request, terse): composite=0.9575  cov=1.00 faith=0.95 act=1.00 tone=0.80
- **refund_request-68967416** (refund_request, frustrated): composite=0.9575  cov=1.00 faith=0.95 act=1.00 tone=0.80
