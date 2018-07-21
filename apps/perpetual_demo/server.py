import datetime, getopt, json, os, socket, sys, time, zmq
from multiprocessing.pool import ThreadPool

import tornado.httpserver
import tornado.websocket
import tornado.tcpserver
import tornado.ioloop
import tornado.web
from tornado import gen
from tornado.iostream import StreamClosedError

g_workers = ThreadPool(2)
g_zmqSubUrl = ""
g_zmqXSubUrl = ""
g_httpPort = 8998
g_wsPort = 8999

class MyWebSocketHandler(tornado.websocket.WebSocketHandler):
  clientConnections = []
  rxCount = 0

  def __init__(self, *args, **kwargs):
    super(MyWebSocketHandler, self).__init__(*args, **kwargs)
    print "[WS] websocket ready"

  def open(self):
    print "[WS] websocket opened"
    MyWebSocketHandler.clientConnections.append(self)

  def on_message(self, message):
    print '[WS] message received %s' % message

  def on_close(self):
    print "[WS] websocket closed"
    MyWebSocketHandler.clientConnections.remove(self)

  @staticmethod
  def broadcastMessage(topic, msg):
    try:
      if not msg:
        return

      msgPOD = {}
      msgPOD['time'] = datetime.datetime.now().strftime("%I:%M:%S.%f %p on %B %d, %Y")
      msgPOD['topic'] = topic
      msgPOD['data'] = msg

      print "[WS] broadcast %s to %d client(s)" \
        % (topic, 
           len(MyWebSocketHandler.clientConnections))
      for socket in MyWebSocketHandler.clientConnections:
        socket.write_message(json.dumps(msgPOD))
    except:
      return

  def check_origin(self, origin):
    return True

wsApp = tornado.web.Application([
  (r"/", MyWebSocketHandler),
])


pwd = os.path.dirname(os.path.realpath(__file__))
httpApp = tornado.web.Application([
  (r"/.*img_val/(.*)", tornado.web.StaticFileHandler,
    {'path': "%s/www/imagenet_val" % pwd } ),
  (r"/static/(.*)", tornado.web.StaticFileHandler,
    {'path': "%s" % pwd} )
])

"""
  Subscribe to C++ system for updates. Send updates to GUI websocket
"""
def backgroundZmqCaffeListener():
  print "Subscribe to C++ updates %s" % g_zmqSubUrl
  context = zmq.Context()
  socket = context.socket(zmq.SUB)
  socket.connect("tcp://%s" % g_zmqSubUrl)
  socket.setsockopt(zmq.SUBSCRIBE, "")

  while True:
    try:
      #  Wait for next request from client
      message = socket.recv()
      #print("Received caffe request: %s" % message)
      MyWebSocketHandler.broadcastMessage("caffe", message)
    except:
      pass

def backgroundZmqXmlrtListener():
  print "Subscribe to xMLrt updates %s" % g_zmqXSubUrl
  context = zmq.Context()
  socket = context.socket(zmq.SUB)
  socket.connect("tcp://%s" % g_zmqXSubUrl)
  socket.setsockopt(zmq.SUBSCRIBE, "")

  while True:
    try:
      #  Wait for next request from client
      message = socket.recv()
      #print("Received xmlrt request: %s" % message)
      MyWebSocketHandler.broadcastMessage("xmlrt", message)
    except:
      pass

def main():
  global g_zmqSubUrl
  global g_zmqXSubUrl

  try:
    opts, args = getopt.getopt(\
      sys.argv[1:], 
      "z:x:", 
      ["zmq_url=", "zmq_xmlrt_url"])
  except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

  for o,a in opts:
    if o == "-z":
      g_zmqSubUrl = a
    elif o == "-x":
      g_zmqXSubUrl = a
  
  print "Start websocket server %s:%d" % (socket.gethostname(), g_wsPort)
  wsServer = tornado.httpserver.HTTPServer(wsApp)
  wsServer.listen(g_wsPort)

  print "Start http server on %s:%d" % (socket.gethostname(), g_httpPort)
  httpServer = tornado.httpserver.HTTPServer(httpApp)
  httpServer.listen(g_httpPort)

  if g_zmqSubUrl:
    g_workers.apply_async(backgroundZmqCaffeListener)
  if g_zmqXSubUrl:
    g_workers.apply_async(backgroundZmqXmlrtListener)

  tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
  main()
