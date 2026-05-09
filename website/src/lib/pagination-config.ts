// Centralized page-size policy for the load-more pagination model.
// Each surface picks an INITIAL (first paint) and INCREMENT (per Load-more click).
// Plain `number` typing (no `as const`) so `useState(PAGE_SIZE.X)` widens to
// `number` rather than narrowing to the literal value.

export const PAGE_SIZE: Readonly<Record<string, number>> = {
  PHOTO_INITIAL: 60,
  PHOTO_INCREMENT: 60,
  EVENT_GRID_INITIAL: 12,
  EVENT_GRID_INCREMENT: 12,
  PORTFOLIO_INITIAL: 9,
  PORTFOLIO_INCREMENT: 9,
  RECEIPT_INITIAL: 10,
  RECEIPT_INCREMENT: 10,
  EARNINGS_INITIAL: 10,
  EARNINGS_INCREMENT: 10,
  TRANSACTION_INITIAL: 25,
  TRANSACTION_INCREMENT: 25,
  STAGED_INITIAL: 50,
  STAGED_INCREMENT: 50,
  ADMIN_INITIAL: 10,
  ADMIN_INCREMENT: 10,
};
