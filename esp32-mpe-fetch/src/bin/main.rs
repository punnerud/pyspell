//! esp32-mpe-fetch — Step 2 coordinator + ST7735 status display.
//!
//! esp-hal 1.1 + esp-rtos (embassy) + esp-radio + embassy-net + async embedded-tls.
//! PySpell programs stay synchronous; the evaluator's I/O is async (`run_async`).
//!
//! - POST `/run`: the body is a PySpell program (Python or Rust syntax). The device
//!   parses it on-device, runs it over the lean stack (verified TLS for `fetch_json`,
//!   host-allowlisted), and returns the result.
//! - Centralized coordinator: a job queue holds only the program text; a small fixed
//!   worker pool drains it, so concurrent TLS memory is bounded no matter the load.
//! - ST7735 shows rolling counters of PySpell jobs completed in the last 10 s / 60 s
//!   / 5 min.

#![no_std]
#![no_main]
#![deny(clippy::mem_forget)]

use core::sync::atomic::{AtomicUsize, Ordering};

use embassy_executor::Spawner;
use embassy_net::tcp::TcpSocket;
use embassy_net::{Config as NetConfig, Runner, Stack, StackResources};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_time::{Duration, Instant, Timer};
use embedded_hal_bus::spi::ExclusiveDevice;
use embedded_io_async::Write as _;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::gpio::{Level, Output, OutputConfig};
use esp_hal::rng::Rng;
use esp_hal::spi::master::{Config as SpiConfig, Spi};
use esp_hal::time::Rate;
use esp_hal::timer::timg::TimerGroup;
use esp_println::println;
use esp_radio::wifi::{sta::StationConfig, Config as WifiConfig, Interface as WifiInterface};

use mipidsi::interface::SpiInterface;
use mipidsi::models::ST7735s;
use mipidsi::options::{ColorInversion, ColorOrder, Orientation, Rotation};
use mipidsi::Builder;

use pyspell_core::{parse, run_async, DslError, EmptyEnv, Lang, Value};

use esp32_mpe_fetch::{config, fetch_async::AsyncLeanNet, fetch_async::TLS_MAX_CONCURRENT, screen};

extern crate alloc;
use alloc::format;
use alloc::string::String;

esp_bootloader_esp_idf::esp_app_desc!();

use esp32_mpe_fetch::{stack_free_now, stack_total};

/// Heap reservation. Sweeping this finds the max heap before the (shrinking) stack
/// can't hold the tailscale registration crypto — i.e. what we actually need.
const HEAP_KB: usize = 160;

const POOL_WORKERS: usize = 2;
const N_SERVERS: usize = 4;
const N_DEMO: usize = 4;
const PORT: u16 = 8080;

/// Default self-test program (the documented weather fetch).
const DEMO_SRC: &str = "fetch_json(\"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=59.91&lon=10.75\", \"properties.timeseries.0.data.instant.details.air_temperature\")";

static DEMO_DONE: AtomicUsize = AtomicUsize::new(0);

/// A queued job: just the PySpell source text (minimal data in the queue).
struct JobReq {
    src: String,
    id: usize,
}

static JOB_QUEUE: Channel<CriticalSectionRawMutex, JobReq, 32> = Channel::new();

macro_rules! mk_static {
    ($t:ty, $val:expr) => {{
        static CELL: static_cell::StaticCell<$t> = static_cell::StaticCell::new();
        CELL.init($val)
    }};
}

#[embassy_executor::task]
async fn net_task(mut runner: Runner<'static, WifiInterface<'static>>) -> ! {
    runner.run().await
}

/// Persistent tailscale control session: register + hold the /machine/map long-poll
/// open so the node stays online. Retries the whole session if the link drops.
#[embassy_executor::task]
async fn ts_task(net: AsyncLeanNet, stack: Stack<'static>) -> ! {
    loop {
        match esp32_mpe_fetch::ts_control::run_control_session(&net, stack).await {
            Ok(()) => println!("[ts] session ended cleanly — reconnecting"),
            Err(e) => println!("[ts] session error: {} — reconnecting", e),
        }
        // The control server closes the map poll after delivering the netmap. Fast
        // reconnect doesn't make the node "online" — that needs the DERP data plane
        // (M3.3); a control-plane-only node has no connection path for peers. So just
        // refresh registration periodically (the IP is stable) without churning the
        // ~58 kB netmap every second.
        Timer::after(Duration::from_secs(30)).await;
    }
}


#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    let hal_config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(hal_config);

    // 128 kB heap (was 192): frees DRAM so the main/executor stack is large enough
    // for tailscale-core's synchronous crypto (snow/x25519/h2) which runs on it.
    esp_alloc::heap_allocator!(size: HEAP_KB * 1024);
    println!(
        "[fetch] === BUILD MARKER R17 (heap={}KB, online-recheck) === heap free {} B; stack total {} B, free here {} B",
        HEAP_KB, esp_alloc::HEAP.free(), stack_total(), stack_free_now()
    );

    // --- ST7735 display on the T-Dongle S3 (SCLK=5 MOSI=3 CS=4 DC=2 RST=1 BL=38) ---
    let cs = Output::new(peripherals.GPIO4, Level::High, OutputConfig::default());
    let dc = Output::new(peripherals.GPIO2, Level::Low, OutputConfig::default());
    let rst = Output::new(peripherals.GPIO1, Level::High, OutputConfig::default());
    // T-Dongle S3 backlight is ACTIVE-LOW: drive low = on. Keep it owned/on.
    let _backlight = Output::new(peripherals.GPIO38, Level::Low, OutputConfig::default());
    let spi_bus = Spi::new(
        peripherals.SPI2,
        SpiConfig::default().with_frequency(Rate::from_mhz(20)),
    )
    .expect("spi")
    .with_sck(peripherals.GPIO5)
    .with_mosi(peripherals.GPIO3);
    let spi_dev = ExclusiveDevice::new(spi_bus, cs, Delay::new()).expect("spi dev");
    let dbuf = mk_static!([u8; 512], [0u8; 512]);
    let di = SpiInterface::new(spi_dev, dc, dbuf);
    let mut init_delay = Delay::new();
    let mut display = Builder::new(ST7735s, di)
        .display_size(80, 160)
        .display_offset(26, 1)
        .invert_colors(ColorInversion::Inverted)
        .color_order(ColorOrder::Bgr)
        .orientation(Orientation::new().rotate(Rotation::Deg90))
        .reset_pin(rst)
        .init(&mut init_delay)
        .expect("display init");
    let _ = screen::draw(&mut display, (esp_alloc::HEAP.free() / 1024) as u32);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_interrupt =
        esp_hal::interrupt::software::SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);
    esp_rtos::start(timg0.timer0, sw_interrupt.software_interrupt0);

    let (mut controller, interfaces) =
        esp_radio::wifi::new(peripherals.WIFI, Default::default()).expect("wifi new");
    controller
        .set_config(&WifiConfig::Station(
            StationConfig::default()
                .with_ssid(config::WIFI_SSID)
                .with_password(config::WIFI_PASS.into()),
        ))
        .expect("set_config");
    println!("[fetch] connecting to '{}' ...", config::WIFI_SSID);
    controller.connect_async().await.expect("wifi connect");
    // Modem sleep between beacons: the radio isn't transmitting full-time, which
    // cuts power/heat at a small latency cost (the device is mostly idle anyway).
    let _ = controller.set_power_saving(esp_radio::wifi::PowerSaveMode::Minimum);
    println!("[fetch] WiFi connected (modem power-save: minimum)");

    let rng = Rng::new();
    let seed = ((rng.random() as u64) << 32) | rng.random() as u64;
    let resources = mk_static!(StackResources<12>, StackResources::new());
    let (stack, runner) = embassy_net::new(
        interfaces.station,
        NetConfig::dhcpv4(Default::default()),
        resources,
        seed,
    );
    spawner.spawn(net_task(runner).expect("spawn net_task"));
    stack.wait_config_up().await;
    if let Some(cfg) = stack.config_v4() {
        println!("[fetch] DHCP lease: {} -> POST PySpell to :{}", cfg.address, PORT);
    }

    let net = AsyncLeanNet::new(stack);

    // --- M3.2: bring up tailscale in a PERSISTENT background task. It registers
    //     (reusing the NVS identity), publishes our IP via TS_IP, then holds the
    //     /machine/map long-poll open forever so the node stays ONLINE. We wait for
    //     the IP here so registration's crypto heap peak passes before the PySpell
    //     workers start; the session then coexists with them on the shared heap. ---
    println!("[ts] registering with tailscale ...");
    spawner.spawn(ts_task(net, stack).expect("spawn ts_task"));
    let ts_ip = esp32_mpe_fetch::ts_control::TS_IP.wait().await;
    println!("[ts] online as {} ; heap {} B free", ts_ip, esp_alloc::HEAP.free());

    for _ in 0..POOL_WORKERS {
        spawner.spawn(worker(net).expect("spawn worker"));
    }

    // Quick boot self-test: submit a few documented weather jobs.
    for id in 0..N_DEMO {
        JOB_QUEUE
            .send(JobReq {
                src: String::from(DEMO_SRC),
                id,
            })
            .await;
    }
    while DEMO_DONE.load(Ordering::Relaxed) < N_DEMO {
        Timer::after(Duration::from_millis(50)).await;
    }
    println!(
        "[fetch] self-test done; POST server up on :{} ({} workers, max {} concurrent TLS)",
        PORT, POOL_WORKERS, TLS_MAX_CONCURRENT
    );
    for id in 0..N_SERVERS {
        spawner.spawn(server_task(id, net).expect("spawn server_task"));
    }

    // Display refresh loop (also keeps the controller alive).
    let _keep_wifi = controller;
    loop {
        let _ = screen::draw(&mut display, (esp_alloc::HEAP.free() / 1024) as u32);
        Timer::after(Duration::from_millis(1000)).await;
    }
}

/// A worker: pull a job off the queue and run its PySpell source.
#[embassy_executor::task(pool_size = POOL_WORKERS)]
async fn worker(net: AsyncLeanNet) {
    loop {
        let req = JOB_QUEUE.receive().await;
        let _ = run_source(net, &req.src, req.id).await;
        DEMO_DONE.fetch_add(1, Ordering::Relaxed);
    }
}

/// Parse PySpell source (Python, else Rust) and run it over the async stack.
/// Records the job for the display counters regardless of outcome.
async fn run_source(net: AsyncLeanNet, src: &str, id: usize) -> Result<Value, DslError> {
    // SAFETY: reject non-ASCII before parsing. PySpell source is ASCII; a
    // multi-byte byte sequence can send the tiny lexer into a loop, and under the
    // cooperative embassy executor an unyielding loop wedges the WHOLE device (a
    // single bad POST took it down in testing). The parse step has no deadline, so
    // we must not let it spin. (A deeper parser fix is a separate pyspell-core TODO.)
    if !src.is_ascii() {
        screen::record_job();
        println!("[job{}] rejected: non-ASCII source", id);
        return Err(DslError::Parse(alloc::string::String::from(
            "only ASCII source is supported",
        )));
    }
    let parsed = parse(src, Lang::Python).or_else(|_| parse(src, Lang::Rust));
    let result = match parsed {
        Ok(mut prog) => {
            prog.max_steps = 200_000;
            let t0 = Instant::now();
            let r = run_async(&prog, &EmptyEnv, Some(&net), prog.max_steps, None, None).await;
            let took = Instant::now().duration_since(t0).as_millis();
            println!("[job{}] ({} ms) -> {:?}", id, took, r);
            r
        }
        Err(e) => {
            println!("[job{}] parse error: {:?}", id, e);
            Err(e)
        }
    };
    screen::record_job();
    println!("[heap] after job{}: free {} B", id, esp_alloc::HEAP.free());
    result
}

/// POST /run acceptor: the request body is a PySpell program; run it and reply.
#[embassy_executor::task(pool_size = N_SERVERS)]
async fn server_task(id: usize, net: AsyncLeanNet) {
    let mut rx = [0u8; 2048];
    let mut tx = [0u8; 1024];
    loop {
        let mut socket = TcpSocket::new(net.stack(), &mut rx, &mut tx);
        socket.set_timeout(Some(Duration::from_secs(20)));
        if socket.accept(PORT).await.is_err() {
            socket.abort();
            Timer::after(Duration::from_millis(50)).await;
            continue;
        }

        // Read the request; the PySpell program is the body after the blank line.
        let mut buf = [0u8; 1536];
        let mut n = 0;
        loop {
            match socket.read(&mut buf[n..]).await {
                Ok(0) => break,
                Ok(k) => {
                    n += k;
                    if find(&buf[..n], b"\r\n\r\n").is_some() || n == buf.len() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let text = core::str::from_utf8(&buf[..n]).unwrap_or("");
        let src = match text.split_once("\r\n\r\n") {
            Some((_, body)) => body.trim(),
            None => "",
        };

        let body = if src.is_empty() {
            String::from("usage: POST a PySpell program as the body, e.g. 1 + 2\n")
        } else {
            match run_source(net, src, 100 + id).await {
                Ok(v) => format!("{:?}\n", v),
                Err(e) => format!("error: {:?}\n", e),
            }
        };
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = socket.write_all(resp.as_bytes()).await;
        let _ = socket.flush().await;
        socket.close();
        Timer::after(Duration::from_millis(20)).await;
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}
