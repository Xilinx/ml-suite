#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/python
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(in_file, 'r') as f:
  lines = f.readlines()

cmds = []
for line in lines:
  line = line.rstrip('\n')

  if line.startswith("#"):
    continue
  
  elif "XNConv" in line:
    words = line.split(" ")
    newWords = []
    for i, word in enumerate(words):
      print i, word
    for i, word in enumerate(words):
      newWords.append(word)
    
      # kernel, stride, in_dim, out_dim:
      if i == 3 or i == 4 or i == 11 or i == 14:
        newWords.append(word)

      if i == 4:
        # add padding
        print words[3]
        padding = int(words[3])/2
        newWords.append(str(padding))
        newWords.append(str(padding))

        # add dilation
        newWords.append("1")
        newWords.append("1")

    # Add te bypass switch
    newWords.append("0")
    cmds.append(' '.join(newWords))

  elif "XNMaxPool" in line:
    words = line.split()
    newWords = []
    for i, word in enumerate(words):
      newWords.append(word)
      if i == 3 or i == 4 or i == 5 or i == 7 or i == 10:
        newWords.append(word)
    newWords.append("0")
    cmds.append(' '.join(newWords))

  elif "XNAvgPool" in line:
    words = line.split()
    newWords = []
    for i, word in enumerate(words):
      newWords.append(word)
      if i == 3 or i == 4 or i == 5 or i == 8 or i == 11:
        newWords.append(word)
    newWords.append("0")
    cmds.append(' '.join(newWords))

  elif "XNEltwise" in line:
    words = line.split()
    newWords = []
    for i, word in enumerate(words):
      newWords.append(word)
      if i == 8:
        newWords.append(word)
    newWords.append("0")
    cmds.append(' '.join(newWords))

  else:
    cmds.append(line)
    
f = open(out_file, 'w')
for cmd in cmds:
  f.write(cmd + "\n")
