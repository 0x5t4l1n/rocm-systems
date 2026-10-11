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
#ifndef LIBRARY_SRC_SDMA_ANVIL_ENGINE_MAP_HPP_
#define LIBRARY_SRC_SDMA_ANVIL_ENGINE_MAP_HPP_

// Pure engine-selection and queue-budget helpers for AnvilLib. Kept free of HSA/KFD headers so the
// unit tests can build them without USE_SDMA.

#include <cstdint>
#include <string>

namespace sdma_anvil {

// Anvil rings every doorbell from a GPU kernel, and an odd-numbered SDMA engine never sees a
// doorbell written that way (ROCM-32598), so a queue there hangs on its first packet. This holds in
// every compute partition mode on MI300X and MI355X, so Anvil places queues on even engines only.
inline bool isUsableSdmaEngine(int engineId) { return engineId >= 0 && engineId % 2 == 0; }

// Even engines of a node reporting numSdmaEnginesTotal engines: 8 on SPX, 4 on DPX, 2 on QPX, 1 on
// CPX. 0 when the counts are unknown.
inline uint32_t usableSdmaEngines(uint32_t numSdmaEnginesTotal) {
  return (numSdmaEnginesTotal + 1) / 2;
}

// Engine to use for an HSA preferred-copy-engine mask: its lowest even engine, else the even engine
// below its lowest odd one. -1 for an empty mask.
inline int preferredUsableEngine(uint32_t preferredMask) {
  if (preferredMask == 0) return -1;
  const uint32_t even = preferredMask & 0x55555555u;
  return even != 0 ? __builtin_ctz(even) : (__builtin_ctz(preferredMask) & ~1);
}

// True when the doubled OAM-map id is past the engines this node reports. numSdmaEnginesTotal == 0
// never folds, so foldOamMapEngine is not asked to divide by zero.
inline bool oamMapEngineNeedsFold(uint32_t numSdmaEnginesTotal, int doubledEngineId) {
  return numSdmaEnginesTotal > 0 && static_cast<uint32_t>(doubledEngineId) >= numSdmaEnginesTotal;
}

// Fold the undoubled OAM-map value and both PCI functions onto the even engines of a node reporting
// numEnginesTotal engines. numEnginesTotal must be > 0. An unreadable function (-1) counts as 0
// and therefore collides with function 0.
inline int foldOamMapEngine(int oamEngine, int srcFn, int dstFn, uint32_t numEnginesTotal) {
  const int src = srcFn < 0 ? 0 : srcFn;
  const int dst = dstFn < 0 ? 0 : dstFn;
  return 2 * ((oamEngine + src + dst) % static_cast<int>(usableSdmaEngines(numEnginesTotal)));
}

// PCI function and the function-0 BDF of the same device. function is -1 when the tail is not a
// function digit, so an unreadable id is not treated as function 0. The physical BDF is rewritten
// only in that case; a bad tail is left unchanged.
struct PciFunctionBus {
  std::string busId;
  std::string physBusId;
  int function;
};

// Split a BDF into its function digit and the function-0 BDF of the same device. Pure, and beside
// the two helpers above because it decides the same fallback they do: a tail outside '0'-'7'
// leaves function at -1 and physBusId equal to busId, which collapses getOamId's candidate list to
// one entry so the physical-BDF read never runs. PCI function numbers are three bits, so '0'-'7'
// is the whole range.
inline PciFunctionBus pciFunctionBus(const std::string& busId) {
  PciFunctionBus loc;
  loc.busId = busId;
  loc.physBusId = busId;
  loc.function = -1;
  if (busId.empty()) return loc;
  const char c = busId.back();
  if (c >= '0' && c <= '7') {
    loc.function = c - '0';
    loc.physBusId.back() = '0';
  }
  return loc;
}

// True when taking `requested` more queues would exceed the queue budget, counted across the usable
// (even) engines as numEngines * queuesPerEngine. Either count being 0 means the budget is unknown, which
// is never reported as exhausted.
inline bool queueBudgetExceeded(uint32_t used, int requested, uint32_t numEngines,
                                uint32_t queuesPerEngine) {
  const uint32_t budget = numEngines * queuesPerEngine;
  return requested > 0 && budget > 0 && used + static_cast<uint32_t>(requested) > budget;
}

}  // namespace sdma_anvil

#endif  // LIBRARY_SRC_SDMA_ANVIL_ENGINE_MAP_HPP_
