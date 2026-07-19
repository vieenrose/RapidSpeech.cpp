#pragma once
// Degenerate-repetition breaker for the MOSS-TD greedy decoder.
//
// LLM-decoder ASR is prone to autoregressive repetition loops (the classic
// failure mode). MOSS greedy decode can cycle on a phrase; the repeats carry
// DIFFERENT timestamps (or none), so an exact-block check misses them. Instead
// we hash the last NGRAM content tokens and track recurrences, but a recurring
// n-gram is only a *loop* under one of three signatures — real speech repeats
// too (councils re-read motions verbatim; statutes enumerate formulas):
//
//   1. tick-stall  — the transcript CLOCK ticks (a [ts] was emitted since the
//                    n-gram's last hit) but does NOT advance >=2 s. A real
//                    segment loop emits a timestamp every cycle that fails to
//                    move forward. Fires at 3 recurrences.
//   2. advancing-clock cycle — a loop that FABRICATES advancing timestamps:
//                    near-constant token stride AND near-constant small ts
//                    delta (<=15 s, within 2 s of the previous delta). Legit
//                    re-reads recur once at an irregular, usually large gap.
//                    Fires at 4 uniform cycles.
//   3. tight loop  — NO timestamp at all since the last hit AND the n-gram
//                    recurs back-to-back (stride <= 16 tokens). Timestamp-free
//                    dense statute reading legitimately repeats every ~50
//                    tokens, so only <=16-token strides, 8+ in a row, count.
//                    (Observed: "這個需求的" x hundreds on a 2 h run.)
//
// On any fire the caller trims the token stream to the first counted
// occurrence (one clean copy survives) and stops decoding.
//
// The class is header-only and free of engine dependencies so it can be unit
// tested against synthetic token/timestamp streams (tests/test_moss_loop_guard.cpp).

#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace rapidspeech {

class MossLoopGuard {
public:
    enum class Reason { None, TickStall, AdvancingCycle, TightLoop };

    struct Decision {
        bool trim_and_stop = false;
        int trim_idx = -1;       // resize the token stream to this length
        Reason reason = Reason::None;
        // diagnostics for the caller's log line
        int count = 0;           // recurrences (tick-stall / tight)
        int stride = 0;
        double delta = 0.0;      // ts delta (advancing cycle)
    };

    // FNV-1a hash of the last NGRAM token ids. Convenience for callers.
    template <typename It>
    static uint64_t hash_ngram(It begin, It end) {
        uint64_t h = 1469598103934665603ULL;
        for (It p = begin; p != end; ++p) {
            h ^= (uint64_t)(uint32_t)(*p);
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Observe one decode step's trailing n-gram.
    //   hash     : hash_ngram over the last NGRAM ids
    //   cur_idx  : number of tokens generated so far (the n-gram's end index)
    //   max_ts   : max timestamp emitted so far (seconds)
    //   ts_ticks : number of timestamps emitted so far
    Decision observe(uint64_t hash, int cur_idx, double max_ts, int ts_ticks) {
        auto it = seen_.find(hash);
        if (it == seen_.end()) {
            seen_.emplace(hash, N{1, cur_idx, max_ts, ts_ticks, cur_idx, -1.0, 1, 0});
            return {};
        }
        N &n = it->second;

        if (max_ts >= n.ts + 2.0) {
            // Clock advanced — usually a legit re-read, unless it recurs with a
            // near-constant stride and near-constant small ts delta.
            const int stride = cur_idx - n.last_idx;
            const double delta = max_ts - n.ts;
            const bool periodic =
                stride <= 256 && delta > 0.0 && delta <= 15.0 &&
                (n.last_delta < 0.0 || std::fabs(delta - n.last_delta) <= 2.0);
            if (periodic && ++n.cyc >= 4) {
                Decision d;
                d.trim_and_stop = true; d.trim_idx = n.first_idx;
                d.reason = Reason::AdvancingCycle; d.count = n.cyc;
                d.stride = stride; d.delta = delta;
                return d;
            }
            const int keep_cyc = periodic ? n.cyc : 1;
            const int keep_first = periodic ? n.first_idx : cur_idx;
            n = N{1, keep_first, max_ts, ts_ticks, cur_idx,
                  periodic ? delta : -1.0, keep_cyc, 0};
            return {};
        }

        if (ts_ticks > n.ts_ticks) {
            // A [ts] was emitted since the last hit but the clock didn't
            // advance >=2 s — stalled clock.
            n.ts_ticks = ts_ticks;
            n.last_idx = cur_idx;
            n.tight = 0;
            if (++n.count >= 3) {
                Decision d;
                d.trim_and_stop = true; d.trim_idx = n.first_idx;
                d.reason = Reason::TickStall; d.count = n.count;
                return d;
            }
            return {};
        }

        // No tick since the last hit. Only a back-to-back (stride <= 16) run
        // counts as a tight loop; sparser timestamp-free repetition is legit.
        const int stride = cur_idx - n.last_idx;
        n.last_idx = cur_idx;
        if (stride > 0 && stride <= 16) {
            if (++n.tight >= 8) {
                Decision d;
                d.trim_and_stop = true; d.trim_idx = n.first_idx;
                d.reason = Reason::TightLoop; d.count = n.tight; d.stride = stride;
                return d;
            }
        } else {
            n.tight = 0;
        }
        return {};
    }

private:
    struct N {
        int count;        // tick-stall recurrences
        int first_idx;    // trim target
        double ts;        // max_ts at (re)start
        int ts_ticks;     // ts count at last hit
        int last_idx;     // token index of last hit
        double last_delta;// last advancing-cycle ts delta (<0 = none)
        int cyc;          // advancing-clock cycles
        int tight;        // consecutive tight (no-ts, small-stride) hits
    };
    std::unordered_map<uint64_t, N> seen_;
};

}  // namespace rapidspeech
