import time
import json
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc, properties=None):
    client.on_connect = on_connect
    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER_URL, 8883)
    print("CONNACK received with code %s." % rc)
  
def on_publish(client, userdata, mid, properties=None):
    client.on_publish = on_publish
    print("Publishing at %s..." %mid)

#dump message
msg = {'id':1, 'x':24.22333, 'y':4.23239, 'people':2, 'item':0, 'time':0.00000}
msgs = [msg,msg]
message = json.dumps({'max_x':50.0000, 'max_y':90.0000 ,'tables': msgs})

client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
client.on_connect()
client.on_publish()
mqtt.client.publish("tables", message, qos=1)
