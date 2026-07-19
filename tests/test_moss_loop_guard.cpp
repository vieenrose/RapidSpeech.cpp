// Unit tests for MossLoopGuard (rapidspeech/src/arch/moss_loop_guard.h).
//
// The guard is the MOSS-TD decoder's degenerate-repetition breaker. These
// tests drive it with synthetic token/timestamp streams reproducing each of
// the three loop signatures plus the legitimate-repetition cases it must NOT
// fire on. No model or audio needed — pure state-machine coverage.

#include "arch/moss_loop_guard.h"

#include <cassert>
#include <cstdio>
#include <vector>

using rapidspeech::MossLoopGuard;
using R = MossLoopGuard::Reason;

namespace {

int g_failures = 0;
#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

// A distinct hash per phrase; the real code hashes the last NGRAM ids, but the
// guard only cares that identical phrases collide and different ones don't.
constexpr uint64_t kLoopHash = 0xABCDEF01u;
constexpr uint64_t kOtherHash = 0x12345678u;

// ---- 1. tight loop: no timestamps, back-to-back (stride <= 16), 8+ times ----
void test_tight_loop_fires() {
    MossLoopGuard g;
    int idx = 100;
    double max_ts = 5.0;         // clock frozen (no [ts] emitted in the loop)
    int ticks = 3;
    MossLoopGuard::Decision fired{};
    for (int i = 0; i < 12; ++i) {
        idx += 13;               // stride 13 <= 16 : tight
        auto d = g.observe(kLoopHash, idx, max_ts, ticks);
        if (d.trim_and_stop) { fired = d; break; }
    }
    CHECK(fired.trim_and_stop, "tight loop should fire");
    CHECK(fired.reason == R::TightLoop, "reason should be TightLoop");
    // first occurrence was at idx 100+13 = 113 (the emplace on hit #1)
    CHECK(fired.trim_idx == 113, "trim to first counted occurrence");
}

// A wide-stride timestamp-free repeat (statute enumeration every ~50 tokens)
// must NOT be treated as a tight loop.
void test_wide_stride_no_fire() {
    MossLoopGuard g;
    int idx = 100;
    for (int i = 0; i < 30; ++i) {
        idx += 50;               // stride 50 > 16 : legitimate re-read cadence
        auto d = g.observe(kLoopHash, idx, 5.0, 3);
        CHECK(!d.trim_and_stop, "wide-stride timestamp-free repeat must not fire");
    }
}

// ---- 2. tick-stall: clock ticks (new [ts]) but never advances >= 2 s ----
void test_tick_stall_fires() {
    MossLoopGuard g;
    int idx = 200;
    int ticks = 0;
    MossLoopGuard::Decision fired{};
    for (int i = 0; i < 6; ++i) {
        idx += 40;               // wide stride, so NOT the tight path
        ++ticks;                 // a [ts] emitted each cycle
        // max_ts inches up < 2 s so the "advanced" branch never triggers
        double max_ts = 10.0 + 0.1 * i;
        auto d = g.observe(kLoopHash, idx, max_ts, ticks);
        if (d.trim_and_stop) { fired = d; break; }
    }
    CHECK(fired.trim_and_stop, "tick-stall loop should fire");
    CHECK(fired.reason == R::TickStall, "reason should be TickStall");
}

// ---- 3. advancing-clock cycle: uniform stride + uniform small ts delta ----
void test_advancing_cycle_fires() {
    MossLoopGuard g;
    int idx = 300;
    double max_ts = 20.0;
    MossLoopGuard::Decision fired{};
    for (int i = 0; i < 8; ++i) {
        idx += 30;               // uniform stride
        max_ts += 5.0;           // uniform +5 s delta, <= 15, per recurrence
        auto d = g.observe(kLoopHash, idx, max_ts, /*ticks*/ 5 + i);
        if (d.trim_and_stop) { fired = d; break; }
    }
    CHECK(fired.trim_and_stop, "advancing-clock cycle should fire");
    CHECK(fired.reason == R::AdvancingCycle, "reason should be AdvancingCycle");
}

// A verbatim re-read that recurs ONCE after a large, irregular time gap is
// legitimate (council re-reading a motion) and must NOT fire.
void test_legit_reread_no_fire() {
    MossLoopGuard g;
    // first occurrence at t=10 s
    auto d1 = g.observe(kLoopHash, 400, 10.0, 5);
    CHECK(!d1.trim_and_stop, "first occurrence never fires");
    // same phrase again 90 s later (real re-read) — single recurrence
    auto d2 = g.observe(kLoopHash, 900, 100.0, 20);
    CHECK(!d2.trim_and_stop, "single large-gap re-read must not fire");
    // and once more, another big irregular gap
    auto d3 = g.observe(kLoopHash, 1500, 210.0, 40);
    CHECK(!d3.trim_and_stop, "second large-gap re-read must not fire");
}

// Varied speech (many distinct n-grams, each recurring only a few times) must
// not fire — the guard keys per phrase, so distinct phrases don't accumulate
// against one another. 10 distinct hashes round-robined for 3 rounds => each
// phrase recurs 3x, below every threshold.
void test_varied_speech_no_fire() {
    MossLoopGuard g;
    int idx = 0;
    for (int round = 0; round < 3; ++round) {
        for (int k = 0; k < 10; ++k) {
            idx += 5;
            uint64_t h = 0x1000u + (uint64_t)k;          // distinct per phrase
            auto d = g.observe(h, idx, 5.0, 3);
            CHECK(!d.trim_and_stop, "varied speech must not fire");
        }
    }
}

}  // namespace

int main() {
    test_tight_loop_fires();
    test_wide_stride_no_fire();
    test_tick_stall_fires();
    test_advancing_cycle_fires();
    test_legit_reread_no_fire();
    test_varied_speech_no_fire();

    if (g_failures == 0) {
        std::printf("test_moss_loop_guard: all checks passed\n");
        return 0;
    }
    std::printf("test_moss_loop_guard: %d check(s) FAILED\n", g_failures);
    return 1;
}
