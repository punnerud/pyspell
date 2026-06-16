//! Phase-1 lwIP bridge (isolation prototype) — present the WireGuard tunnel as a
//! real lwIP network interface so the device's std `TcpListener` serves Tailscale
//! traffic with a *real* TCP stack (retransmit, windows, parallel connections),
//! instead of the single-connection, no-retransmit toy server in `core/src/tcp.rs`.
//!
//! This is the "WG netif bridge" already scoped in `router.rs` /
//! `docs/PLAN-router-exitnode.md` (option 2), narrowed to: serve our *own* 100.x.
//! No NAPT / IP_FORWARD needed for that.
//!
//! ## Why an Ethernet-type netif (and the ARP shim)
//! Only the `eth` netstack is compiled into this build (no PPP/SLIP/tun). So the WG
//! netif is an Ethernet interface: injected packets need a 14-byte L2 header, the
//! `transmit` buffer is a full Ethernet frame, and lwIP ARPs for next hops. Since
//! `etharp_add_static_entry` isn't bound, we answer ARP ourselves with a fictional
//! peer MAC (the L2 is meaningless for a tunnel — we route by IP on egress).
//!
//! ## Data flow
//! - inbound: `inject_ip(decrypted_inner)` → prepend eth header → `esp_netif_receive`.
//! - outbound: the `wg_transmit` callback (esp_netif thread) answers ARP, or strips
//!   the eth header and pushes the inner IP packet onto [`drain_tx`] for the data
//!   plane to encrypt + send (encryption stays on the data-plane thread, which owns
//!   the `wg::Tunnel`s).
//!
//! Step 1a here is the **isolation loopback test** ([`loopback_selftest`]): inject one
//! ICMP echo request and confirm an ARP request + an ICMP echo reply come back out
//! the `transmit` callback over serial — proving the netif end-to-end before any
//! data-plane integration.

#![allow(clippy::missing_safety_doc)]

use std::collections::VecDeque;
use std::net::Ipv4Addr;
use std::sync::Mutex;

use esp_idf_svc::nvs::{EspDefaultNvsPartition, EspNvs};
use esp_idf_svc::sys;

// --- L2 constants for the fictional Ethernet framing of the tunnel ---
const ETH_HDR: usize = 14;
const ETHERTYPE_IP: u16 = 0x0800;
const ETHERTYPE_ARP: u16 = 0x0806;
/// Our netif's MAC (locally-administered, unicast).
const OUR_MAC: [u8; 6] = [0x02, 0x54, 0x53, 0x00, 0x00, 0x01];
/// The single fictional "peer" MAC we answer all ARP queries with.
const PEER_MAC: [u8; 6] = [0x02, 0x54, 0x53, 0x00, 0x00, 0x02];

/// Outgoing inner IP packets captured from lwIP, awaiting encrypt+send by the data
/// plane. Bounded so a stall can't grow unbounded.
static TX_QUEUE: Mutex<VecDeque<Vec<u8>>> = Mutex::new(VecDeque::new());
const TX_QUEUE_MAX: usize = 64;

/// The WG netif handle, stored once created (single interface).
static NETIF: Mutex<usize> = Mutex::new(0);

/// Frames handed to lwIP via `esp_netif_receive` are referenced **zero-copy**
/// (PBUF_REF), so they must outlive lwIP's use and be freed only when lwIP is done
/// (via our `driver_free_rx_buffer` callback). Keep each boxed frame alive here,
/// keyed by its data pointer, until [`wg_free_rx`] reclaims it. (Freeing the Rust
/// buffer early — the original bug — caused a tcpip_thread use-after-free crash.)
static RX_BUFS: Mutex<Vec<(usize, Box<[u8]>)>> = Mutex::new(Vec::new());

/// Hand an Ethernet `frame` to lwIP (zero-copy), keeping it alive in `RX_BUFS`.
unsafe fn esp_inject(netif: *mut sys::esp_netif_t, frame: Vec<u8>) {
    let boxed: Box<[u8]> = frame.into_boxed_slice();
    let len = boxed.len();
    let ptr = boxed.as_ptr() as usize;
    RX_BUFS.lock().unwrap().push((ptr, boxed));
    let buf = ptr as *mut core::ffi::c_void;
    // eb == buf: esp_netif passes it back to wg_free_rx when lwIP frees the pbuf.
    sys::esp_netif_receive(netif, buf, len, buf);
}

/// esp_netif `driver_free_rx_buffer`: lwIP finished with an injected frame — drop it.
unsafe extern "C" fn wg_free_rx(_h: *mut core::ffi::c_void, eb: *mut core::ffi::c_void) {
    let key = eb as usize;
    let mut v = RX_BUFS.lock().unwrap();
    if let Some(pos) = v.iter().position(|(p, _)| *p == key) {
        v.swap_remove(pos);
    }
}

fn netif_ptr() -> *mut sys::esp_netif_t {
    *NETIF.lock().unwrap() as *mut sys::esp_netif_t
}

/// Drain the outgoing inner IP packets lwIP has produced. The data plane calls this
/// each tick, maps each packet's dst-100.x to a tunnel, and encrypts + sends it.
/// (Wired into the data plane in the next Phase-1 step; unused in the Step-1a test.)
#[allow(dead_code)]
pub fn drain_tx() -> Vec<Vec<u8>> {
    let mut q = TX_QUEUE.lock().unwrap();
    q.drain(..).collect()
}

/// Inject a decrypted inner IPv4 packet from the tunnel into lwIP (prepends the
/// fictional Ethernet header). Safe to call from the data-plane thread —
/// `esp_netif_receive` posts to the tcpip thread internally.
pub fn inject_ip(ip_packet: &[u8]) {
    let netif = netif_ptr();
    if netif.is_null() || ip_packet.is_empty() {
        return;
    }
    let mut frame = Vec::with_capacity(ETH_HDR + ip_packet.len());
    frame.extend_from_slice(&OUR_MAC); // dst = us
    frame.extend_from_slice(&PEER_MAC); // src = the peer
    frame.extend_from_slice(&ETHERTYPE_IP.to_be_bytes());
    frame.extend_from_slice(ip_packet);
    unsafe { esp_inject(netif, frame) }
}

/// The esp_netif driver `transmit` callback: lwIP hands us a full Ethernet frame to
/// "send". We answer ARP locally; for IP we strip the L2 header and queue the inner
/// packet for the data plane.
unsafe extern "C" fn wg_transmit(
    _h: *mut core::ffi::c_void,
    buffer: *mut core::ffi::c_void,
    len: usize,
) -> sys::esp_err_t {
    if buffer.is_null() || len < ETH_HDR {
        return sys::ESP_OK as sys::esp_err_t;
    }
    let frame = core::slice::from_raw_parts(buffer as *const u8, len);
    let ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    match ethertype {
        ETHERTYPE_ARP => handle_arp(&frame[ETH_HDR..]),
        ETHERTYPE_IP => {
            let ip = &frame[ETH_HDR..];
            println!(
                "lwip_bridge: TX {} bytes IP -> {}",
                ip.len(),
                dst_ip(ip).map(|a| a.to_string()).unwrap_or_default()
            );
            let mut q = TX_QUEUE.lock().unwrap();
            if q.len() < TX_QUEUE_MAX {
                q.push_back(ip.to_vec());
            }
        }
        other => println!("lwip_bridge: TX ethertype {other:#06x} ({len} B) ignored"),
    }
    sys::ESP_OK as sys::esp_err_t
}

/// Read the IPv4 destination address from an inner IP packet.
fn dst_ip(ip: &[u8]) -> Option<Ipv4Addr> {
    if ip.len() >= 20 && (ip[0] >> 4) == 4 {
        Some(Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19]))
    } else {
        None
    }
}

/// Answer an ARP request (opcode 1) for any target IP with our fictional peer MAC,
/// by injecting an ARP reply back up the stack. ARP payload layout (Ethernet/IPv4):
/// htype(2) ptype(2) hlen(1) plen(1) op(2) sha(6) spa(4) tha(6) tpa(4).
unsafe fn handle_arp(arp: &[u8]) {
    if arp.len() < 28 {
        return;
    }
    let op = u16::from_be_bytes([arp[6], arp[7]]);
    if op != 1 {
        return; // only answer requests
    }
    let sender_mac = &arp[8..14]; // requester (lwIP / us)
    let sender_ip = &arp[14..18];
    let target_ip = &arp[24..28]; // who lwIP is asking about
    println!(
        "lwip_bridge: ARP who-has {}.{}.{}.{} -> reply peer MAC",
        target_ip[0], target_ip[1], target_ip[2], target_ip[3]
    );

    // Build the ARP reply payload: target answers, addressed back to the sender.
    let mut payload = Vec::with_capacity(28);
    payload.extend_from_slice(&[0x00, 0x01]); // htype Ethernet
    payload.extend_from_slice(&ETHERTYPE_IP.to_be_bytes()); // ptype IPv4
    payload.push(6); // hlen
    payload.push(4); // plen
    payload.extend_from_slice(&[0x00, 0x02]); // op = reply
    payload.extend_from_slice(&PEER_MAC); // sha = peer
    payload.extend_from_slice(target_ip); // spa = the asked-for IP
    payload.extend_from_slice(sender_mac); // tha = back to requester
    payload.extend_from_slice(sender_ip); // tpa

    // Wrap in an Ethernet frame addressed to the requester and inject.
    let netif = netif_ptr();
    if netif.is_null() {
        return;
    }
    let mut frame = Vec::with_capacity(ETH_HDR + payload.len());
    frame.extend_from_slice(sender_mac); // dst = requester
    frame.extend_from_slice(&PEER_MAC); // src = peer
    frame.extend_from_slice(&ETHERTYPE_ARP.to_be_bytes());
    frame.extend_from_slice(&payload);
    esp_inject(netif, frame);
}

/// A custom esp_netif IO driver. `base` MUST be the first field: esp_netif reads
/// the driver through the handle as an `esp_netif_driver_base_t*` to call
/// `post_attach`. `Box::leak`'d so it lives for the netif's lifetime.
#[repr(C)]
struct WgDriver {
    base: sys::esp_netif_driver_base_t,
    ifconfig: sys::esp_netif_driver_ifconfig_t,
}

/// Called by `esp_netif_attach`: record our netif and register the driver ifconfig
/// (the transmit callback). This is the correct attach lifecycle — the earlier
/// `set_driver_config` + fake-handle shortcut caused an InstrFetchProhibited panic.
unsafe extern "C" fn wg_post_attach(
    netif: *mut sys::esp_netif_t,
    h: sys::esp_netif_iodriver_handle,
) -> sys::esp_err_t {
    let drv = h as *mut WgDriver;
    (*drv).base.netif = netif;
    (*drv).ifconfig.handle = h; // transmit() receives the driver as `h` (unused)
    sys::esp_netif_set_driver_config(netif, &(*drv).ifconfig)
}

/// Create and bring up the WG netif with our Tailscale `ip` (mask /10 = the
/// 100.64.0.0/10 CGNAT range tailnet addresses live in). Returns Ok once lwIP owns
/// the address; a `TcpListener` bound to 0.0.0.0 then accepts connections on it.
///
/// NOTE: the esp_netif custom-driver attach/start lifecycle is the part most likely
/// to need a device-driven tweak — keep an eye on serial during bring-up.
pub fn create(ip: Ipv4Addr) -> Result<(), sys::esp_err_t> {
    unsafe {
        // Base on the default eth inherent config, then drop DHCP (we set a static
        // IP) and give it our own key/MAC.
        let mut base: sys::esp_netif_inherent_config_t = sys::_g_esp_netif_inherent_eth_config;
        base.flags &= !sys::esp_netif_flags_ESP_NETIF_DHCP_CLIENT;
        base.if_key = c"WG_TS".as_ptr();
        base.if_desc = c"tailscale".as_ptr();
        base.mac = OUR_MAC;
        base.ip_info = core::ptr::null(); // set explicitly below

        let cfg = sys::esp_netif_config_t {
            base: &base,
            driver: core::ptr::null(),
            stack: sys::_g_esp_netif_netstack_default_eth,
        };
        let netif = sys::esp_netif_new(&cfg);
        if netif.is_null() {
            return Err(sys::ESP_FAIL as sys::esp_err_t);
        }

        // Proper custom-driver attach: a driver-base whose post_attach wires the
        // transmit callback. Box::leak so it outlives this fn (netif holds it).
        let drv: &'static mut WgDriver = Box::leak(Box::new(WgDriver {
            base: sys::esp_netif_driver_base_t {
                post_attach: Some(wg_post_attach),
                netif: core::ptr::null_mut(),
            },
            ifconfig: sys::esp_netif_driver_ifconfig_t {
                handle: core::ptr::null_mut(), // set in post_attach
                transmit: Some(wg_transmit),
                transmit_wrap: None,
                // REQUIRED: lwIP calls this to free each injected (zero-copy) frame.
                // Leaving it None was the InstrFetchProhibited crash (null fn call).
                driver_free_rx_buffer: Some(wg_free_rx),
            },
        }));
        let err = sys::esp_netif_attach(
            netif,
            drv as *mut WgDriver as sys::esp_netif_iodriver_handle,
        );
        if err != sys::ESP_OK as i32 {
            return Err(err);
        }

        // Static IP (mask 255.192.0.0 = /10, no gateway — peers are "on-link").
        let ip_info = sys::esp_netif_ip_info_t {
            ip: u32_to_ip4(ip),
            netmask: sys::esp_ip4_addr_t {
                addr: u32::from(Ipv4Addr::new(255, 192, 0, 0)).to_be(),
            },
            gw: sys::esp_ip4_addr_t { addr: 0 },
        };
        let _ = sys::esp_netif_dhcpc_stop(netif); // ignore "already stopped"
        let err = sys::esp_netif_set_ip_info(netif, &ip_info);
        if err != sys::ESP_OK as i32 {
            return Err(err);
        }

        // Mark started + link up so the stack passes packets.
        sys::esp_netif_action_start(
            netif as *mut core::ffi::c_void,
            core::ptr::null_mut(),
            0,
            core::ptr::null_mut(),
        );
        sys::esp_netif_action_connected(
            netif as *mut core::ffi::c_void,
            core::ptr::null_mut(),
            0,
            core::ptr::null_mut(),
        );

        *NETIF.lock().unwrap() = netif as usize;
        println!("lwip_bridge: WG netif up at {ip}/10");
        Ok(())
    }
}

fn u32_to_ip4(ip: Ipv4Addr) -> sys::esp_ip4_addr_t {
    // esp_ip4_addr.addr is little-endian-on-wire u32 (network order in memory).
    sys::esp_ip4_addr_t {
        addr: u32::from(ip).to_be(),
    }
}

/// Boot-loop SAFEGUARD wrapper around the (still experimental) bring-up. An NVS
/// counter is bumped *before* the risky FFI runs and reset only after the device has
/// proven stable for a while. If a bad attach panics and reboots, the counter climbs;
/// once it hits `MAX_TRIES` we skip the bring-up entirely, so the device always comes
/// back green within a couple of boots instead of bricking into a USB-JTAG-blocking
/// loop. Recovery if it ever does loop: hold BOOT + replug + release, then
/// `espflash flash --before no-reset <elf>`.
pub fn bring_up_guarded(part: &EspDefaultNvsPartition, ours: Ipv4Addr) {
    const NS: &str = "lwipbr";
    const KEY: &str = "tries";
    const MAX_TRIES: u8 = 2;

    let nvs = match EspNvs::new(part.clone(), NS, true) {
        Ok(n) => n,
        Err(e) => {
            println!("lwip_bridge: NVS open failed ({e}); skipping bring-up");
            return;
        }
    };
    let mut buf = [0u8; 1];
    let tries = nvs
        .get_blob(KEY, &mut buf)
        .ok()
        .flatten()
        .and_then(|b| b.first().copied())
        .unwrap_or(0);

    if tries >= MAX_TRIES {
        println!("lwip_bridge: SAFEGUARD — disabled after {tries} failed boots; clearing counter (reflash/clear to retry)");
        let _ = nvs.set_blob(KEY, &[0]);
        return;
    }
    // Bump BEFORE the risky FFI so a panic-reboot is counted.
    let _ = nvs.set_blob(KEY, &[tries + 1]);
    println!("lwip_bridge: bring-up attempt {}/{}", tries + 1, MAX_TRIES);

    match create(ours) {
        Ok(()) => {
            // Serve our 100.x over real TCP via the lwIP netif (replaces the toy
            // tcp.rs server for tunnel HTTP). ONE worker = one TCP served at a time:
            // lowest memory, and no contention on the single-core dataplane drain
            // (two concurrent connections starved each other + leaked lwIP PCBs).
            // Extra connections queue (graceful pause) and are served in turn.
            let _ = std::thread::Builder::new()
                .stack_size(4096)
                .spawn(|| crate::local_server::run_port(80, 1, 20 * 1024, true));
            println!("lwip_bridge: HTTP server on :80 (WG netif, real TCP, serialized)");
        }
        Err(e) => println!("lwip_bridge: create failed (err={e})"),
    }

    // Survived the risky section — clear the counter after a stability window so a
    // transient issue doesn't permanently disable, but a real panic-loop self-disables.
    let part2 = part.clone();
    let _ = std::thread::Builder::new()
        .stack_size(3072)
        .spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(30));
            if let Ok(n) = EspNvs::new(part2, NS, true) {
                let _ = n.set_blob(KEY, &[0]);
                println!("lwip_bridge: stable 30s — safeguard counter reset");
            }
        });
}

/// Step-1a isolation test: inject one ICMP echo request from `peer` to `ours` and
/// rely on serial logs from [`wg_transmit`] to show the ARP request + echo reply.
/// Retired now that the dataplane drives inject/drain, but kept for debugging.
#[allow(dead_code)]
pub fn loopback_selftest(ours: Ipv4Addr, peer: Ipv4Addr) {
    println!("lwip_bridge: loopback self-test — inject ICMP echo {peer} -> {ours}");
    let pkt = build_icmp_echo_request(peer, ours, 0x1234, 1);
    inject_ip(&pkt);
}

/// Build a minimal IPv4 + ICMP echo-request packet (8-byte ICMP, no payload).
#[allow(dead_code)]
fn build_icmp_echo_request(src: Ipv4Addr, dst: Ipv4Addr, id: u16, seq: u16) -> Vec<u8> {
    let mut icmp = vec![8u8, 0, 0, 0]; // type=8 (echo), code=0, csum placeholder
    icmp.extend_from_slice(&id.to_be_bytes());
    icmp.extend_from_slice(&seq.to_be_bytes());
    let csum = checksum(&icmp);
    icmp[2..4].copy_from_slice(&csum.to_be_bytes());

    let total = 20 + icmp.len();
    let mut ip = vec![0u8; 20];
    ip[0] = 0x45; // v4, ihl=5
    ip[2..4].copy_from_slice(&(total as u16).to_be_bytes());
    ip[8] = 64; // TTL
    ip[9] = 1; // proto ICMP
    ip[12..16].copy_from_slice(&src.octets());
    ip[16..20].copy_from_slice(&dst.octets());
    let ipcsum = checksum(&ip);
    ip[10..12].copy_from_slice(&ipcsum.to_be_bytes());
    ip.extend_from_slice(&icmp);
    ip
}

/// Standard one's-complement Internet checksum.
#[allow(dead_code)]
fn checksum(data: &[u8]) -> u16 {
    let mut sum = 0u32;
    let mut i = 0;
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}
