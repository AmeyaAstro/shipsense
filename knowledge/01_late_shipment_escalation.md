# Late-Shipment Escalation Procedure

> **FICTIONAL DOCUMENT.** Synthetic policy written for the ShipSense portfolio project. "ShipSense Logistics" is not a real company and none of this is real operational guidance.

## Purpose
Defines who acts, and how fast, when an order is late or is predicted to be late.

## Escalation tiers
- **Tier 1 (1 day past scheduled delivery):** the Customer Care agent assigned to the order
  contacts the customer within 4 business hours and logs the case in the ticket system.
- **Tier 2 (2–3 days late):** the case moves to the Regional Operations Lead, who must contact the
  carrier and post an updated delivery estimate within 1 business day.
- **Tier 3 (4 or more days late, or any order above $2,000 that is 2+ days late):** the Logistics
  Duty Manager owns the case, a refund or credit review is opened automatically, and the customer
  receives a call, not only an email.

## Predicted-late orders
Orders flagged `is_late_risk = true` by the ShipSense model are handled under the High-Risk Order
Handling policy before they are late. If a flagged order then misses its date, it enters Tier 1
directly; it does not restart the clock.

## Closing a case
A case may be closed only after delivery is confirmed and the customer has been told about any
credit they are owed. Cases are reviewed weekly; any Tier 3 case open more than 10 business days
is reported to the Head of Operations.
