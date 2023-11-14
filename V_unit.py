# incomplete code here
# Use python to call model files of unitv2 pretrained models

from json.decoder import JSONDecodeError
import subprocess
import json
import base64

reconizer = subprocess.Popen(['/home/m5stack/payload/bin/object_recognition',
                              '/home/m5stack/payload/uploads/models/nanodet_80class'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

reconizer.stdin.write("_{\"stream\":1}\r\n".encode('utf-8'))
reconizer.stdin.flush()

img = b''

while 1:
    doc = json.loads(reconizer.stdout.readline().decode('utf-8'))
    print(doc)
    if 'img' in doc:
        byte_data = base64.b64decode(doc["img"])
        img = bytes(byte_data)
    elif 'num' in doc:
        for obj in doc['obj']:
            print(obj)
