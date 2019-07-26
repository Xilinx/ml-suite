#!/usr/bin/python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import curses
import os
import re
import sys
import time

class Screen:
  def __enter__(self):
    self.stdscr = curses.initscr()
    curses.cbreak()
    curses.noecho()
    self.stdscr.keypad(1)
    SCREEN_HEIGHT, SCREEN_WIDTH = self.stdscr.getmaxyx()
    return self.stdscr

  def __exit__(self,a,b,c):
    curses.nocbreak()
    self.stdscr.keypad(0)
    curses.echo()
    curses.endwin()

class Stats:
  def __init__(self):
    self.vals = []
    self.vrange = None
    self.max_buffer = []
    self.winsize = 50

  def add(self, v):
    self.vals.append(v)
    self.vals = self.vals[-self.winsize:]
    self.vrange = [v,v] if self.vrange == None \
      else [min(self.vrange[0], v), max(self.vrange[1], v)]

  def mean(self):
    if len(self.vals) == 0:
      return 0.

    return sum(self.vals) / len(self.vals)

  def min(self, dynamic=False):
    if self.vrange == None:
      return 0.

    if dynamic:
      return min(self.vals)

    return self.vrange[0]

  def max(self, dynamic=False):
    if self.vrange == None:
      return 0.

    if dynamic:
      return max(self.vals)

    return self.vrange[1]
  
  def n(self):
    return len(self.vals)

def getBar(val, maxVal, c):
  if not maxVal:
    return ''
  pct = val/maxVal 
  return c * int(pct*6)

def printLine(screen, row, line):
  h, w = screen.getmaxyx()
  screen.addstr(row, 0, line.ljust(w))

def printStats(screen, stats, numPEs):
  rightSpace = " " * 30 
  maxMean = 0
  for k in stats:
    if k in ["latency", "exec", "format", "input"]:
      continue

    v = stats[k].mean()
    if k == "quant" and "format" in stats:
      # merge 'quant' and 'format'
      v += stats["format"].mean()
    
    maxMean = max(maxMean, v)

  printLine(screen, 0, "XDNN pipeline report")
  printLine(screen, 1, "--------------------")

  ycurse = 3
  toPrint = ["quant", "ddr_wr", "submit", "fpga_0", "fpga_1", 
             "ddr_rd", "post", "done"]
  for k in toPrint:
    if k not in stats:
      continue

    mean = stats[k].mean()
    minv = stats[k].min()
    pctBar = getBar(mean, maxMean, '|')

    label = k
    if k == "quant" and "format" in stats:
      label = "quant+format"
      fmean = stats["format"].mean()
      mean += fmean
      minv += stats["format"].min()
      pctBar += getBar(fmean, maxMean, '/')

    slowestStageFlag = ''
    if mean == maxMean:
      slowestStageFlag = 'x'

    printLine(screen, ycurse,
      "%13s %8s %6s (min: %.2f) %s" \
      % (label, pctBar, "%.2f" % mean, minv, slowestStageFlag))
    ycurse += 1

  inputRate = 0
  fpgaRate = 0
  efficiency = 0

  if "input" in stats:
    minv = stats["input"].min(dynamic=True)
    maxv = stats["input"].max(dynamic=True)
    period = float(maxv - minv) / 1000000.
    n = float(stats["input"].n())

    if n and period:
      inputRate = n/period

  if maxMean:
    fpgaRate = numPEs * 1000 / maxMean

  if fpgaRate:
    efficiency = min(inputRate / fpgaRate * 100., 100.)

  printLine(screen, ycurse, " ")
  ycurse += 1
  printLine(screen, ycurse,
    "Input rate          : %d images/s" % int(inputRate))
  ycurse += 1
  printLine(screen, ycurse,
    "Max FPGA throughput : %d images/s with %d PEs "
    "(pre-/post-processing not included)" \
    % (int(fpgaRate), numPEs))
  ycurse += 1
  printLine(screen, ycurse, 
    "FPGA utilization    : %.2f%%" % efficiency)

  if "latency" in stats:
    oversubbedMsg = "                                   "
    try:
      fpgaLatency = stats["fpga_0"].mean() + stats["fpga_1"].mean()
      if stats["exec"].mean() > fpgaLatency:
        pct = (stats["exec"].mean() - fpgaLatency) * 100 / fpgaLatency
        if pct > 10:
          oversubbedMsg = "(FPGA is %d%% oversubscribed)" % int(pct)
        else:
          oversubbedMsg = "(FPGA performance is input-limited)"
    except:
      pass 

    ycurse += 1
    printLine(screen, ycurse,
      "End-to-end latency  : %.2f ms %s" \
        % (stats["latency"].mean(), oversubbedMsg))

  screen.refresh()

keys = [
  ("quant", "quant"),
  ("prep", "format"),
  ("ddr_wr", "ddr_wr"),
  ("submit", "submit"),
  ("hw_counter_0", "fpga_0"),
  ("hw_counter_1", "fpga_1"),
  ("exec", "exec"),
  ("ddr_rd", "ddr_rd"),
  ("post", "post"),
  (": =", "latency")]

if __name__ == '__main__':
  stats = {}
  numPEs = 0
  snapshotNum = 0

  with Screen() as screen:
    for l in sys.stdin:
      if "[XDNN]" not in l:
        continue
      line = l.rstrip()

      if snapshotNum < 100 and "FPGA metrics" in line:
        # auto-detect num PEs
        try:
          numPEs = max(int(line.split("/")[1]) + 1, numPEs)
        except:
          pass
        continue  

      match = re.match(r".+ Packet \d+: (.+)", line)
      if match:
        m = match.group(1)

        if m[0].isdigit():
          # packet timestamp
          ts = int(m)
          if "input" not in stats:
            stats["input"] = Stats()
          stats["input"].add(ts)

        if m.startswith("="):
          # end of packet 
          snapshotNum += 1

      if snapshotNum % 50 != 0:
        # don't slam the CPU, sample only 2%
        continue

      for i, (k, pk) in enumerate(keys):
        if k not in line:
          continue

        valStr = line.split(k)[1]
        match = re.match(r".* (\d+\.\d+).*", valStr)
        if not match:
          continue
        val = float(match.group(1))

        if pk not in stats:
          stats[pk] = Stats()

        stats[pk].add(val)

        break 

      printStats(screen, stats, numPEs)

  time.sleep(10)

