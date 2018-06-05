#!/usr/bin/python
import os, sys

path = os.getcwd()

lookingFor = "MLsuite"
while True:
  words = path.split("/")
  if len(words) <= 1:
    break

  leaf = words[-1]
  if leaf == lookingFor:
    # found root
    print(path)
    sys.exit(0)

  path = os.path.dirname(path)

# not found
print("")
