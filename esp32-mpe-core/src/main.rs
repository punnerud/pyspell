//! esp32-mpe-core baseline: measure free SRAM (heap) on the pure-Rust esp-rs
//! stack, before and after bringing up WiFi — the comparison point against
//! esp-idf's ~60 kB-free figure. Flash footprint is read from the built image.
//!
//! Prints once at boot over USB-Serial-JTAG, then idles.

#![no_std]
#![no_main]

extern crate alloc;

use esp_backtrace as _;
use esp_hal::rng::Rng;
use esp_hal::timer::timg::TimerGroup;
use esp_println::println;

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());

    // Reserve a generous heap so WiFi's allocations come out of it and we can
    // see how much remains. (The chip has ~320 kB usable internal DRAM.)
    esp_alloc::heap_allocator!(192 * 1024);
    println!("[mpe-core] heap free at boot:          {} B", esp_alloc::HEAP.free());

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let rng = Rng::new(peripherals.RNG);

    let wifi_ctrl = esp_wifi::init(timg0.timer0, rng, peripherals.RADIO_CLK)
        .expect("esp_wifi::init failed");
    println!(
        "[mpe-core] heap free after wifi init:  {} B   (wifi used: {} B)",
        esp_alloc::HEAP.free(),
        esp_alloc::HEAP.used()
    );

    // Creating the STA device allocates the rx/tx queues a real connection uses.
    let _ = esp_wifi::wifi::new_with_mode(
        &wifi_ctrl,
        peripherals.WIFI,
        esp_wifi::wifi::WifiStaDevice,
    );
    println!(
        "[mpe-core] heap free after STA device: {} B   (wifi used: {} B)",
        esp_alloc::HEAP.free(),
        esp_alloc::HEAP.used()
    );
    println!("[mpe-core] DONE — compare to esp-idf ~60 kB free (wifi+tailscale+pyspell)");

    // Re-print periodically so a serial capture can catch the numbers any time.
    let delay = esp_hal::delay::Delay::new();
    loop {
        println!(
            "[mpe-core] HEAP free={} B used={} B",
            esp_alloc::HEAP.free(),
            esp_alloc::HEAP.used()
        );
        delay.delay_millis(2000);
    }
}
