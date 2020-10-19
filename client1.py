from websocket import create_connection
import logging
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(' %(module)s -  %(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
 
ws = create_connection("ws://127.0.0.1:12345")
ws.send("1")
result = ws.recv()
logger.info("Received '{}'".format(result))
ws.close()
logger.info("Close")