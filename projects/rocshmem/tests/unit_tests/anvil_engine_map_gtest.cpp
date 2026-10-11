/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "gtest/gtest.h"
#include "../src/sdma/anvil_engine_map.hpp"

using namespace sdma_anvil;

namespace {

// MI300X engine counts as ROCr reports them, per logical GPU.
constexpr uint32_t kSpxTotal = 16;
constexpr uint32_t kDpxTotal = 8;
constexpr uint32_t kQpxTotal = 4;
constexpr uint32_t kCpxTotal = 2;

}  // namespace

TEST(AnvilEngineMap, OnlyEvenEnginesAreUsable) {
  // Odd engines never see a GPU-written doorbell (ROCM-32598).
  for (int e = 0; e < static_cast<int>(kSpxTotal); ++e) {
    EXPECT_EQ(isUsableSdmaEngine(e), e % 2 == 0) << "engine=" << e;
  }
  EXPECT_FALSE(isUsableSdmaEngine(-1));
}

TEST(AnvilEngineMap, UsableEnginesAreTheEvenHalf) {
  EXPECT_EQ(usableSdmaEngines(kSpxTotal), 8u);
  EXPECT_EQ(usableSdmaEngines(kDpxTotal), 4u);
  EXPECT_EQ(usableSdmaEngines(kQpxTotal), 2u);
  EXPECT_EQ(usableSdmaEngines(kCpxTotal), 1u);
  EXPECT_EQ(usableSdmaEngines(0), 0u);
}

TEST(AnvilEngineMap, PreferredMaskPicksAnEvenEngine) {
  EXPECT_EQ(preferredUsableEngine(0x0004), 2);
  EXPECT_EQ(preferredUsableEngine(0x0006), 2);  // engines 1 and 2
  EXPECT_EQ(preferredUsableEngine(0x0002), 0);  // engine 1 alone
  EXPECT_EQ(preferredUsableEngine(0x8000), 14);
  EXPECT_EQ(preferredUsableEngine(0), -1);
}

TEST(AnvilEngineMap, SpxUsesInRangeIdsAsIs) {
  for (int id = 0; id < static_cast<int>(kSpxTotal); id += 2) {
    EXPECT_FALSE(oamMapEngineNeedsFold(kSpxTotal, id)) << "id=" << id;
  }
}

TEST(AnvilEngineMap, OutOfRangeIdFolds) {
  EXPECT_TRUE(oamMapEngineNeedsFold(kSpxTotal, static_cast<int>(kSpxTotal)));
  EXPECT_TRUE(oamMapEngineNeedsFold(kDpxTotal, 12));
}

TEST(AnvilEngineMap, UnknownTotalNeverFolds) {
  EXPECT_FALSE(oamMapEngineNeedsFold(0, 0));
  EXPECT_FALSE(oamMapEngineNeedsFold(0, 12));
}

TEST(AnvilEngineMap, FoldLandsOnAnInRangeEvenEngine) {
  for (uint32_t total : {kDpxTotal, kQpxTotal, kCpxTotal}) {
    for (int oam = 0; oam < 8; ++oam) {
      for (int src = -1; src < 8; ++src) {
        for (int dst = -1; dst < 8; ++dst) {
          const int e = foldOamMapEngine(oam, src, dst, total);
          EXPECT_TRUE(isUsableSdmaEngine(e)) << "total=" << total << " e=" << e;
          EXPECT_LT(e, static_cast<int>(total)) << "total=" << total;
        }
      }
    }
  }
}

TEST(AnvilEngineMap, CpxFoldAlwaysUsesEngineZero) {
  for (int oam = 0; oam < 8; ++oam) {
    EXPECT_EQ(foldOamMapEngine(oam, 3, 5, kCpxTotal), 0);
  }
}

TEST(AnvilEngineMap, FoldSeparatesPairsSharingADestinationFunction) {
  EXPECT_NE(foldOamMapEngine(0, 0, 1, kDpxTotal), foldOamMapEngine(0, 1, 1, kDpxTotal));
}

TEST(AnvilEngineMap, FoldTreatsUnreadableFunctionAsZero) {
  // One unreadable side: without the clamp, (0 + -1 + 0) % 4 would be -1.
  EXPECT_EQ(foldOamMapEngine(0, -1, 0, kDpxTotal), foldOamMapEngine(0, 0, 0, kDpxTotal));
  EXPECT_EQ(foldOamMapEngine(0, 0, -1, kDpxTotal), foldOamMapEngine(0, 0, 0, kDpxTotal));
  EXPECT_EQ(foldOamMapEngine(3, -1, -1, kDpxTotal), foldOamMapEngine(3, 0, 0, kDpxTotal));
}

TEST(AnvilEngineMap, PciFunctionBusRewritesPartitionToFunctionZero) {
  const PciFunctionBus loc = pciFunctionBus("0000:05:00.1");
  EXPECT_EQ(loc.function, 1);
  EXPECT_EQ(loc.busId, "0000:05:00.1");
  EXPECT_EQ(loc.physBusId, "0000:05:00.0");
}

TEST(AnvilEngineMap, PciFunctionBusAcceptsOnlyThreeBitFunctions) {
  for (char c = '0'; c <= '7'; ++c) {
    EXPECT_EQ(pciFunctionBus(std::string("0000:05:00.") + c).function, c - '0');
  }
  for (const char* bad : {"0000:05:00.8", "0000:05:00.9", "0000:05:00.a"}) {
    const PciFunctionBus loc = pciFunctionBus(bad);
    EXPECT_EQ(loc.function, -1) << bad;
    EXPECT_EQ(loc.physBusId, loc.busId) << bad;
  }
}

TEST(AnvilEngineMap, QueueBudgetRefusesOverSubscription) {
  // 2 engines x 8 queues per engine.
  EXPECT_FALSE(queueBudgetExceeded(9, 7, 2, 8));
  EXPECT_TRUE(queueBudgetExceeded(10, 7, 2, 8));
  EXPECT_TRUE(queueBudgetExceeded(0, 17, 2, 8));
}

TEST(AnvilEngineMap, QueueBudgetCountsEvenEnginesOnly) {
  // CPX, 8 queues per engine: engine 0 alone, so 8 queues at 1 channel but only 4 peers at 2.
  const uint32_t cpx = usableSdmaEngines(kCpxTotal);
  EXPECT_FALSE(queueBudgetExceeded(7, 1, cpx, 8));
  EXPECT_TRUE(queueBudgetExceeded(8, 1, cpx, 8));
  EXPECT_FALSE(queueBudgetExceeded(6, 2, cpx, 8));
  EXPECT_TRUE(queueBudgetExceeded(8, 2, cpx, 8));
  // QPX: engines 0 and 2, 16 queues, so 32 ranks (31 peers) at 1 channel do not fit.
  const uint32_t qpx = usableSdmaEngines(kQpxTotal);
  EXPECT_FALSE(queueBudgetExceeded(15, 1, qpx, 8));
  EXPECT_TRUE(queueBudgetExceeded(16, 1, qpx, 8));
}

TEST(AnvilEngineMap, QueueBudgetUnknownNeverRefuses) {
  EXPECT_FALSE(queueBudgetExceeded(100, 8, 0, 8));
  EXPECT_FALSE(queueBudgetExceeded(100, 8, kCpxTotal, 0));
}

TEST(AnvilEngineMap, QueueBudgetIgnoresEmptyRequest) {
  EXPECT_FALSE(queueBudgetExceeded(16, 0, kCpxTotal, 8));
  EXPECT_FALSE(queueBudgetExceeded(16, -1, kCpxTotal, 8));
}

TEST(AnvilEngineMap, PciFunctionBusHandlesEmptyInput) {
  const PciFunctionBus loc = pciFunctionBus("");
  EXPECT_EQ(loc.function, -1);
  EXPECT_TRUE(loc.physBusId.empty());
}
