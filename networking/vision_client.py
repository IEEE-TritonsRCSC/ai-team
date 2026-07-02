"""
SSLVisionClient: drop-in replacement for `sslclient.client()` that fixes a
multi-NIC bug in the upstream `sslclient` package (v1.1.2).

Upstream's `connect()` joins the vision multicast group using
`socket.gethostbyname(socket.gethostname())` to pick the local interface.
That call does not reliably return the IP of a specific wired NIC — on a
machine with both Wi-Fi and a directly-plugged vision ethernet cable, it can
resolve to the Wi-Fi address (or loopback), which joins the multicast group
on the wrong interface. The socket then sits idle with no error: vision
packets simply never arrive.

This class reimplements the same connect/receive logic, but joins on an
explicit interface IP (`SSL_NETWORK_INTERFACE` from net_config.py) when one
is configured, instead of guessing via gethostbyname/gethostname.
"""

import socket

from sslclient.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket

from .net_config import SSL_VISION_MULTICAST_IP, SSL_VISION_PORT, SSL_NETWORK_INTERFACE


class SSLVisionClient:
    def __init__(
        self,
        ip: str = SSL_VISION_MULTICAST_IP,
        port: int = SSL_VISION_PORT,
        interface: str | None = SSL_NETWORK_INTERFACE,
    ):
        self.ip = ip
        self.port = port
        self.interface = interface
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.bind((self.ip, self.port))

        # Explicit interface if configured (the field vision NIC, e.g. the
        # directly-plugged ethernet), otherwise fall back to the same
        # hostname-resolution guess upstream sslclient uses.
        iface_ip = self.interface or socket.gethostbyname(socket.gethostname())
        self.sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface_ip))
        self.sock.setsockopt(
            socket.SOL_IP, socket.IP_ADD_MEMBERSHIP,
            socket.inet_aton(self.ip) + socket.inet_aton(iface_ip),
        )

    def receive(self) -> SSL_WrapperPacket:
        data, _ = self.sock.recvfrom(1024)
        return SSL_WrapperPacket().FromString(data)
