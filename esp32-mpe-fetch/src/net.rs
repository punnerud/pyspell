//! Synchronous network stack: raw smoltcp driven over esp-radio's Wi-Fi
//! `Interface`, with DHCP + DNS. No embassy-net, no async (the only async we
//! touch is `connect_async`, bridged with `block_on` in `main.rs`).
//!
//! esp-radio 0.18 only ships an `embassy-net-driver` impl, so we wrap its public
//! `Interface::{receive,transmit}` + `Wifi{Rx,Tx}Token::consume_token` as a
//! `smoltcp::phy::Device` ourselves (the pre-embassy esp-wifi pattern).

use alloc::vec::Vec;

use embedded_io::{ErrorKind, ErrorType, Read as IoRead, Write as IoWrite};
use esp_radio::wifi::{Interface as WifiInterface, WifiRxToken, WifiTxToken};
use smoltcp::iface::{Config as IfaceConfig, Interface, SocketHandle, SocketSet};
use smoltcp::phy::{Device, DeviceCapabilities, Medium, RxToken, TxToken};
use smoltcp::socket::{dhcpv4, dns, tcp};
use smoltcp::time::Instant as SmolInstant;
use smoltcp::wire::{
    DnsQueryType, EthernetAddress, HardwareAddress, IpAddress, IpCidr, Ipv4Address,
};

/// Wi-Fi MTU (Ethernet payload).
const MTU: usize = 1500;

/// Monotonic milliseconds since boot, as a smoltcp `Instant`.
pub fn smol_now() -> SmolInstant {
    let ms = esp_hal::time::Instant::now()
        .duration_since_epoch()
        .as_millis();
    SmolInstant::from_millis(ms as i64)
}

// --- smoltcp phy::Device over esp-radio's Wi-Fi Interface ---------------------

pub struct WifiSmolDevice<'d> {
    iface: WifiInterface<'d>,
}

pub struct SmolRxToken(WifiRxToken);
pub struct SmolTxToken(WifiTxToken);

impl RxToken for SmolRxToken {
    fn consume<R, F: FnOnce(&[u8]) -> R>(self, f: F) -> R {
        // esp-radio hands us `&mut [u8]`; smoltcp only needs `&[u8]`.
        self.0.consume_token(|buf| f(buf))
    }
}

impl TxToken for SmolTxToken {
    fn consume<R, F: FnOnce(&mut [u8]) -> R>(self, len: usize, f: F) -> R {
        self.0.consume_token(len, f)
    }
}

impl Device for WifiSmolDevice<'_> {
    type RxToken<'a>
        = SmolRxToken
    where
        Self: 'a;
    type TxToken<'a>
        = SmolTxToken
    where
        Self: 'a;

    fn receive(&mut self, _ts: SmolInstant) -> Option<(Self::RxToken<'_>, Self::TxToken<'_>)> {
        self.iface
            .receive()
            .map(|(rx, tx)| (SmolRxToken(rx), SmolTxToken(tx)))
    }

    fn transmit(&mut self, _ts: SmolInstant) -> Option<Self::TxToken<'_>> {
        self.iface.transmit().map(SmolTxToken)
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.max_transmission_unit = MTU;
        caps.medium = Medium::Ethernet;
        caps
    }
}

// --- The synchronous IP stack -------------------------------------------------

pub struct LeanStack<'d> {
    device: WifiSmolDevice<'d>,
    iface: Interface,
    sockets: SocketSet<'static>,
    dhcp: SocketHandle,
    dns: SocketHandle,
}

impl<'d> LeanStack<'d> {
    /// Build the stack over a *connected* Wi-Fi station interface.
    pub fn new(station: WifiInterface<'d>, seed: u64) -> Self {
        let mut device = WifiSmolDevice { iface: station };
        let mac = device.iface.mac_address();

        let mut config = IfaceConfig::new(HardwareAddress::Ethernet(EthernetAddress(mac)));
        config.random_seed = seed;

        let iface = Interface::new(config, &mut device, smol_now());

        let mut sockets = SocketSet::new(Vec::new());
        let dhcp = sockets.add(dhcpv4::Socket::new());
        // No DNS servers yet — filled in from the DHCP lease.
        let dns = sockets.add(dns::Socket::new(&[], Vec::new()));

        Self {
            device,
            iface,
            sockets,
            dhcp,
            dns,
        }
    }

    /// One poll of the IP stack.
    pub fn poll(&mut self) {
        self.iface
            .poll(smol_now(), &mut self.device, &mut self.sockets);
    }

    /// Busy-poll until we have a DHCP lease (or `deadline_ms` elapses). Returns
    /// the leased IPv4 address, the gateway, and how many DNS servers we learned.
    pub fn run_dhcp(&mut self, deadline_ms: u64) -> Option<(Ipv4Address, Option<Ipv4Address>, usize)> {
        let start = esp_hal::time::Instant::now();
        loop {
            self.poll();
            // Copy the owned bits out of the borrowed DHCP `Config`, then drop the
            // socket borrow before touching `iface`/`sockets` again.
            let configured = match self.sockets.get_mut::<dhcpv4::Socket>(self.dhcp).poll() {
                Some(dhcpv4::Event::Configured(cfg)) => {
                    let servers: Vec<IpAddress> = cfg
                        .dns_servers
                        .iter()
                        .map(|a| IpAddress::Ipv4(*a))
                        .collect();
                    Some((cfg.address, cfg.router, servers))
                }
                _ => None,
            };
            if let Some((cidr, router, servers)) = configured {
                self.iface.update_ip_addrs(|addrs| {
                    addrs.clear();
                    let _ = addrs.push(IpCidr::Ipv4(cidr));
                });
                if let Some(r) = router {
                    let _ = self.iface.routes_mut().add_default_ipv4_route(r);
                }
                let n = servers.len();
                self.sockets
                    .get_mut::<dns::Socket>(self.dns)
                    .update_servers(&servers);
                return Some((cidr.address(), router, n));
            }
            if start.elapsed().as_millis() > deadline_ms {
                return None;
            }
        }
    }

    /// Resolve `host` to an IPv4 address via DNS (busy-poll, with a deadline).
    pub fn resolve(&mut self, host: &str, deadline_ms: u64) -> Option<IpAddress> {
        let Self {
            iface,
            sockets,
            dns,
            ..
        } = self;
        let query = {
            let cx = iface.context();
            sockets
                .get_mut::<dns::Socket>(*dns)
                .start_query(cx, host, DnsQueryType::A)
                .ok()?
        };

        let start = esp_hal::time::Instant::now();
        loop {
            self.poll();
            match self
                .sockets
                .get_mut::<dns::Socket>(self.dns)
                .get_query_result(query)
            {
                Ok(addrs) => return addrs.first().copied(),
                Err(dns::GetQueryResultError::Pending) => {}
                Err(_) => return None,
            }
            if start.elapsed().as_millis() > deadline_ms {
                return None;
            }
        }
    }

    /// Open a TCP connection to `remote:port` (busy-poll until established).
    /// Returns the socket handle on success.
    pub fn connect_tcp(
        &mut self,
        remote: IpAddress,
        port: u16,
        local_port: u16,
        rx_bytes: usize,
        tx_bytes: usize,
        deadline_ms: u64,
    ) -> Option<SocketHandle> {
        let rx = tcp::SocketBuffer::new(alloc::vec![0u8; rx_bytes]);
        let tx = tcp::SocketBuffer::new(alloc::vec![0u8; tx_bytes]);
        let handle = self.sockets.add(tcp::Socket::new(rx, tx));

        {
            let Self { iface, sockets, .. } = self;
            let cx = iface.context();
            let sock = sockets.get_mut::<tcp::Socket>(handle);
            if sock.connect(cx, (remote, port), local_port).is_err() {
                return None;
            }
        }

        let start = esp_hal::time::Instant::now();
        loop {
            self.poll();
            let sock = self.sockets.get_mut::<tcp::Socket>(handle);
            match sock.state() {
                tcp::State::Established => return Some(handle),
                tcp::State::Closed => return None,
                _ => {}
            }
            if start.elapsed().as_millis() > deadline_ms {
                return None;
            }
        }
    }

    /// Wrap a connected TCP socket as a blocking embedded-io transport (for TLS).
    pub fn tcp_conn(&mut self, handle: SocketHandle, deadline_ms: u64) -> TcpConn<'_, 'd> {
        TcpConn {
            stack: self,
            handle,
            deadline_ms,
        }
    }
}

// --- Blocking embedded-io transport over a smoltcp TCP socket -----------------

/// Errors from the blocking TCP transport.
#[derive(Debug)]
pub enum TcpError {
    Recv,
    Send,
    Closed,
    Timeout,
}

impl core::fmt::Display for TcpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl core::error::Error for TcpError {}

impl embedded_io::Error for TcpError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::Other
    }
}

/// A blocking `embedded_io::{Read,Write}` view of one TCP socket, driving the
/// smoltcp poll loop on each call. Borrows the whole stack for the session.
pub struct TcpConn<'s, 'd> {
    stack: &'s mut LeanStack<'d>,
    handle: SocketHandle,
    deadline_ms: u64,
}

impl ErrorType for TcpConn<'_, '_> {
    type Error = TcpError;
}

impl IoRead for TcpConn<'_, '_> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, TcpError> {
        let start = esp_hal::time::Instant::now();
        loop {
            self.stack.poll();
            let sock = self.stack.sockets.get_mut::<tcp::Socket>(self.handle);
            if sock.can_recv() {
                return sock.recv_slice(buf).map_err(|_| TcpError::Recv);
            }
            if !sock.may_recv() {
                return Ok(0); // remote closed → EOF
            }
            if start.elapsed().as_millis() > self.deadline_ms {
                return Err(TcpError::Timeout);
            }
        }
    }
}

impl IoWrite for TcpConn<'_, '_> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, TcpError> {
        let start = esp_hal::time::Instant::now();
        loop {
            self.stack.poll();
            let sock = self.stack.sockets.get_mut::<tcp::Socket>(self.handle);
            if sock.can_send() {
                return sock.send_slice(buf).map_err(|_| TcpError::Send);
            }
            if !sock.may_send() {
                return Err(TcpError::Closed);
            }
            if start.elapsed().as_millis() > self.deadline_ms {
                return Err(TcpError::Timeout);
            }
        }
    }

    fn flush(&mut self) -> Result<(), TcpError> {
        let start = esp_hal::time::Instant::now();
        loop {
            self.stack.poll();
            let sock = self.stack.sockets.get_mut::<tcp::Socket>(self.handle);
            if sock.send_queue() == 0 {
                return Ok(());
            }
            if start.elapsed().as_millis() > self.deadline_ms {
                return Err(TcpError::Timeout);
            }
        }
    }
}
