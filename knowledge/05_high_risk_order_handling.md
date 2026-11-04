# High-Risk Order Handling

> **FICTIONAL DOCUMENT.** Synthetic policy written for the ShipSense portfolio project. "ShipSense Logistics" is not a real company and none of this is real operational guidance.

## What "high risk" means
The ShipSense model returns a late probability for each order at the time it is placed. An order
is **high risk** when `is_late_risk = true`, i.e. its probability is at or above the model's
decision threshold. The threshold is tuned on validation data and published with each model
version; do not hard-code it in other systems.

## Required actions for high-risk orders
1. **Proactive notice:** email the customer a realistic delivery window within 2 business hours of
   the order being placed.
2. **Upgrade check:** if the order value is above $500 and it ships Standard Class, the
   Fulfilment Lead may upgrade it to Second Class at our cost (see Shipping Mode Upgrade Policy).
3. **Tag the order** with HIGH_RISK in the order system so Customer Care sees it.

## What not to do
- Do not cancel or delay an order because it is high risk.
- Do not tell customers the model's exact probability; give a delivery window instead.

## Model limits
The model ranks orders reasonably well (test ROC-AUC about 0.78) but is not certain. Treat a flag
as a reason to act early, not as a guarantee the order will be late.
