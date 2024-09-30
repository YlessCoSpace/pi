import socket
import netifaces as ni
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed


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


# Run the scan
if __name__ == '__main__':
    open_rtsp_ips = scan_ip_port(8554, max_workers=255)
    print(f'Devices with RTSP (port 8554) open: {open_rtsp_ips}')
