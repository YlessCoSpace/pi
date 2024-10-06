import cv2
import socket
import netifaces as ni
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
import paho.mqtt.client as mqtt


def scan_ip_port(port, *, max_workers=255) -> list[str]:
    def get_network_info():
        iface = ni.gateways()['default'][ni.AF_INET][1]
        ip_info = ni.ifaddresses(iface)[ni.AF_INET][0]
        return ip_info['addr'], ip_info['netmask']

    def get_ip_range():
        local_ip, subnet_mask = get_network_info()
        return ipaddress.IPv4Network(f'{local_ip}/{subnet_mask}', strict=False)

    def scan_ip(_ip, _port, _timeout):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(_timeout)

        try:
            _result = sock.connect_ex((str(_ip), _port))
            if _result == 0:
                return str(_ip)
        except socket.error:
            pass
        finally:
            sock.close()

        return None

    network = get_ip_range()
    open_ips = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ip = {executor.submit(scan_ip, ip, port, 0.5): ip for ip in network.hosts()}

        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                result = future.result()
                if result:
                    open_ips.append(result)
            except Exception as exc:
                print(f'IP {ip} generated an exception: {exc}')

    return open_ips


def find_network_cam(username: str, password: str) -> cv2.VideoCapture:
    cam_ip = scan_ip_port(8554, max_workers=255)
    if cam_ip:
        return cv2.VideoCapture(f'rtsp://{username}:{password}@{cam_ip[0]}:8554/live')
    else:
        raise FileNotFoundError('No video devices on the network')


class MQTTPublisher:
    def __init__(self, broker_address, username=None, password=None, port=1883):
        self.broker_address = broker_address
        self.port = port
        self.client = mqtt.Client()

        if username and password:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_address, self.port, 60)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print(f'Connected to {self.broker_address} with result code {rc}')

    def publish(self, topic, message):
        self.client.publish(topic, message)
        print(f'Published `{message}` to `{topic}`')

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print('Disconnected from the broker.')


__all__ = [
    'scan_ip_port',
    'find_network_cam',
    'MQTTPublisher'
]
