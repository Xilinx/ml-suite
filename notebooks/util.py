def plot_all(X):
  import numpy as np 
  import matplotlib.pyplot as plt
  
  ax = plt.subplot(221)
  ax.set_title("Vector X in FP32 Representation",fontsize=20)
  plt.setp(ax.get_xticklabels(), visible=False)
  ax.set_ylabel("Value")
  plt.plot(X["fp32"])

  bx = plt.subplot(222)
  bx.set_title("Mapping FP32 to INT8",fontsize=20)
  plt.setp(bx.get_xticklabels(), visible=False)
  bx.set_ylabel("Value")
  plt.plot(X["fp32"])
  plt.plot(X["threshold"]*np.ones_like(X["fp32"]),"k",linewidth=5.0)
  plt.plot(-1*X["threshold"]*np.ones_like(X["fp32"]),"k",linewidth=5.0)

  for i in range(128):
    plt.plot(i/X["sf"]*np.ones_like(X["fp32"]),"m",linewidth=0.1)
        
  for i in range(128):
    plt.plot(-1*i/X["sf"]*np.ones_like(X["fp32"]),"m",linewidth=0.1)

  cx = plt.subplot(223)
  cx.set_title("Vector X in INT8 Representation",fontsize=20)
  cx.set_xlabel("Element #")
  cx.set_ylabel("Value")
  plt.plot(X["int8"])

  dx = plt.subplot(224)
  dx.set_title("Percent Error by Element",fontsize=20)
  dx.set_xlabel("Element #")
  dx.set_ylabel("Percent Error")
  plt.plot(X["perror"])

  plt.subplots_adjust(top=2,bottom=0.1,left=0.1,right=4,wspace=0.2)

def plot_all2(X):
  import numpy as np 
  import matplotlib.pyplot as plt
  
  ax = plt.subplot(221)
  ax.set_title("Vector Y in FP32 Representation",fontsize=20)
  plt.setp(ax.get_xticklabels(), visible=False)
  ax.set_ylabel("Value")
  plt.plot(X["fp32"][0])

  """
  bx = plt.subplot(222)
  bx.set_title("Mapping FP32 to INT8",fontsize=20)
  plt.setp(bx.get_xticklabels(), visible=False)
  bx.set_ylabel("Value")
  plt.plot(X["fp32"][0])
  """

  cx = plt.subplot(223)
  cx.set_title("Vector Y in INT Representation",fontsize=20)
  cx.set_xlabel("Element #")
  cx.set_ylabel("Value")
  plt.plot(X["int"][0])

  dx = plt.subplot(224)
  dx.set_title("Percent Error by Element",fontsize=20)
  dx.set_xlabel("Element #")
  dx.set_ylabel("Percent Error")
  plt.plot(X["perror"][0])

  plt.subplots_adjust(top=2,bottom=0.1,left=0.1,right=4,wspace=0.2)

def findShiftScale(val):

  import numpy as np
  # val = x * 2^e
  # e must be a negative integer
  # x must be a positive integer
  e = np.ceil(np.log2(val))
  x = 1

  e_lifo = []
  x_lifo = []

  approx = x * 2**e
  delta = val-approx
  oldloss = np.square(val-approx)
  
  while True:
    approx = x * 2**e
    delta = val-approx
    loss = np.square(val-approx)

    if loss < oldloss and delta > 0:
      e_lifo.append(e)
      x_lifo.append(x)

    oldloss = loss

    if delta < 0: # Make approximation smaller
      e -= 1
      x *= 2
      x -= 1

    else:
      x += 1

    if x > 256 or e < -40:
      return e_lifo[-1],x_lifo[-1]
