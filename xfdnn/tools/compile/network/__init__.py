import os, sys
for d in ["codegeneration","graph","memory","network","optimizations", "weights","version","tests"]:
  path = "%s/../%s" % (os.path.dirname(os.path.realpath(__file__)), d)
  sys.path.insert(0, path)
