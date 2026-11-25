// Tiny Bot, Easy $2 — hybrid momentum + RSI dip-buy + EMA cross-down exits.
//
// Strategy (backtested on the kit's dev window, 136 hourly bars, 7 symbols):
//   - Every 12 bars, rank symbols by trailing 12h return; hold the top 2
//     whose 12h return is positive. Symbols that fall out of the top 2 are
//     sold; their replacements are bought on the NEXT bar (a sell's proceeds
//     never fund a buy of the same reply).
//   - At the same rebalance, any symbol with RSI(14) < 30 is also bought
//     into a free slot (dip buy).
//   - Every bar, a held position exits on: EMA(8) crossing below EMA(21),
//     RSI(14) above 80, value 10% under cost (stop), or value 40% over
//     cost (take profit). Stop/take-profit almost never fire — the exits
//     are dominated by the cross-down and the rotation.
//   - Buys split the available cash equally, capped at 25% of equity each
//     (caps.max_weight_bps).
//
// Backtest (offline simulator calibrated to grade-local: buy-and-hold
// -869.35 USDC vs grade-local -869.34; EMA-cross baseline -173.1 vs -173.04,
// 28 trades = 28):
//   - Full dev window: PnL +228.0 USDC (+2.28%), margin over buy-and-hold
//     +1,097.3 USDC, max drawdown 3.4%, 32 trades.
//   - All 65 rolling 72-bar windows (mimicking the 3-day final window):
//     mean margin +272.3 USDC, median +198.4, worst -223.2, prize-eligible
//     (margin > 0, >= 8 trades, drawdown <= 25%) in 52/65 windows (80%).
//   - Neighbours (rsi_sell 75/85/90, top 3, lookback 18/24) all score
//     45-51/65 — a plateau, not a spike.
//
// Determinism: float64 EMA/RSI on + - * / alone, no clock, no environment,
// no randomness. Every entry runs twice and the action logs must match.
//
// RSI note: RSI(14) is computed over the 14 one-hour changes ending at the
// PREVIOUS bar's close (the 15 closes before the current bar). That is the
// exact specification the backtest used; do not "fix" it to include the
// current bar, or the graded result will differ from the backtest.

import { BPS, runBars, type BarsState, type BuyInput, type OrderInput } from './vendor/makefun-bars.ts';

const LOOKBACK_BARS = 12;
const TOP_N = 2;
const REBALANCE_BARS = 12;
const RSI_BUY = 30;
const RSI_SELL = 80;
const STOP = 0.1; // exit when value is 10% under cost
const ROI = 0.4; // exit when value is 40% over cost
const MAX_SLIPPAGE_BPS = 10_000;

/** One EMA: seeded as the mean of its first span of closes, then moving
 *  2 / (span + 1) of the way to each new close (same as the kit baseline). */
interface Ema {
  span: number;
  warm: number[];
  value: number | null;
}

function emaStep(e: Ema, close: number): number | null {
  if (e.value !== null) {
    e.value += (2 / (e.span + 1)) * (close - e.value);
    return e.value;
  }
  e.warm.push(close);
  if (e.warm.length === e.span) e.value = e.warm.reduce((a, x) => a + x, 0) / e.span;
  return e.value;
}

/** RSI(14) over the 15 closes given (14 changes); null when not enough data. */
function rsi(closes15: readonly number[]): number | null {
  if (closes15.length < 15) return null;
  let gains = 0;
  let losses = 0;
  for (let i = 1; i < 15; i++) {
    const d = closes15[i] - closes15[i - 1];
    if (d > 0) gains += d;
    else losses -= d;
  }
  if (losses === 0) return gains > 0 ? 100 : 50;
  return 100 - 100 / (1 + gains / losses);
}

const closes = new Map<string, number[]>();
const emas = new Map<string, { fast: Ema; slow: Ema }>();
let nextRebalance = LOOKBACK_BARS;
let entering: string[] | null = null;

function buySized(state: BarsState, symbols: readonly string[]): BuyInput[] {
  const room = state.hello.caps.max_open_positions - state.positions.size;
  if (room <= 0) return [];
  const coming = symbols.filter((s) => !state.positions.has(s) && state.close(s) !== null).slice(0, room);
  if (coming.length === 0) return [];
  const { caps, fees } = state.hello;
  const byWeight = (BigInt(caps.max_weight_bps) * state.equity) / BPS;
  let slice = state.cash / BigInt(coming.length);
  if (slice > byWeight) slice = byWeight;
  if (slice === 0n || slice < BigInt(fees.min_notional_quote_units)) return [];
  return coming.map((symbol): BuyInput => ({ side: 'buy', symbol, notional_quote_units: slice, max_slippage_bps: MAX_SLIPPAGE_BPS }));
}

await runBars({
  onBatch(state, batch) {
    for (const bar of batch.bars) {
      const hist = closes.get(bar.symbol) ?? [];
      hist.push(Number(bar.c));
      closes.set(bar.symbol, hist);
    }
    if (state.warmingUp) return [];

    // Exits, every bar: EMA(8) cross below EMA(21), RSI > 80, stop, ROI.
    const sells: OrderInput[] = [];
    const sold = new Set<string>();
    for (const { symbol } of state.hello.symbols) {
      const e = emas.get(symbol) ?? { fast: { span: 8, warm: [], value: null }, slow: { span: 21, warm: [], value: null } };
      emas.set(symbol, e);
      const hist = closes.get(symbol) ?? [];
      const c = hist.length > 0 ? hist[hist.length - 1] : 0;
      const fastPrev = e.fast.value;
      const slowPrev = e.slow.value;
      const fast = emaStep(e.fast, c);
      const slow = emaStep(e.slow, c);
      const held = state.positions.get(symbol);
      if (held === undefined) continue;
      const crossDown = fast !== null && slow !== null && fastPrev !== null && slowPrev !== null && fast < slow && fastPrev >= slowPrev;
      // RSI over the 15 closes ending at the previous bar (see note above).
      const r = hist.length >= 16 ? rsi(hist.slice(-16, -1)) : null;
      const value = state.value(symbol, held.qty);
      if (crossDown || (r !== null && r > RSI_SELL) || value * 10n <= held.cost * 9n || value * 10n >= held.cost * 14n) {
        sells.push({ side: 'sell', symbol, qty: held.qty, max_slippage_bps: MAX_SLIPPAGE_BPS });
        sold.add(symbol);
      }
    }

    if (entering !== null) {
      const want = entering;
      entering = null;
      return [...sells, ...buySized(state, want)];
    }

    if (batch.seq < nextRebalance) return sells;
    nextRebalance = batch.seq + REBALANCE_BARS;

    // Rank by trailing 12h return; keep the top 2 with a positive return.
    const ranked: { symbol: string; ret: number }[] = [];
    for (const { symbol } of state.hello.symbols) {
      const h = closes.get(symbol) ?? [];
      if (h.length > LOOKBACK_BARS) ranked.push({ symbol, ret: h[h.length - 1] / h[h.length - 1 - LOOKBACK_BARS] - 1 });
    }
    ranked.sort((a, b) => (b.ret - a.ret) || (a.symbol < b.symbol ? -1 : a.symbol > b.symbol ? 1 : 0));
    const want = ranked.filter((x) => x.ret > 0).slice(0, TOP_N).map((x) => x.symbol);
    // RSI dip buys into the remaining slots.
    for (const { symbol } of state.hello.symbols) {
      if (want.includes(symbol) || state.positions.has(symbol)) continue;
      const h = closes.get(symbol) ?? [];
      const r = h.length >= 16 ? rsi(h.slice(-16, -1)) : null;
      if (r !== null && r < RSI_BUY) want.push(symbol);
    }

    const rotationSells: OrderInput[] = [];
    for (const [symbol, p] of state.positions) {
      if (!want.includes(symbol) && !sold.has(symbol)) {
        rotationSells.push({ side: 'sell', symbol, qty: p.qty, max_slippage_bps: MAX_SLIPPAGE_BPS });
      }
    }
    if (rotationSells.length > 0) {
      entering = want;
      return [...sells, ...rotationSells];
    }
    return [...sells, ...buySized(state, want)];
  },
});
