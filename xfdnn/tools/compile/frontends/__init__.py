
# FIXME: make these behave like Python module subdirs as well
import os, sys
for d in ["codegeneration","graph","memory","network","optimizations", "weights","version","bin"]:
  path = "%s/../%s" % (os.path.dirname(os.path.realpath(__file__)), d)
  print path
  sys.path.insert(0, path)
