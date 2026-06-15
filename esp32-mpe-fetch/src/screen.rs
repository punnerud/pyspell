//! ST7735 status display + rolling PySpell job counters.
//!
//! Each completed PySpell job calls [`record_job`]; the display shows how many
//! completed in the last 10 s / 60 s / 5 min. The counter is a small ring of
//! completion timestamps behind a critical section (workers record, the display
//! loop reads), pruned to the 5-minute window.

use core::cell::RefCell;
use core::fmt::Write as _;

use critical_section::Mutex;
use embedded_graphics::mono_font::ascii::{FONT_6X10, FONT_9X15_BOLD};
use embedded_graphics::mono_font::MonoTextStyle;
use embedded_graphics::pixelcolor::Rgb565;
use embedded_graphics::prelude::*;
use embedded_graphics::text::Text;
use heapless::{Deque, String};

const WIN_10S: u64 = 10_000;
const WIN_60S: u64 = 60_000;
const WIN_5MIN: u64 = 300_000;
const WIN_60MIN: u64 = 3_600_000;
/// Ring of recent completion timestamps. At >~1024 jobs/60min the 60min count
/// saturates (oldest dropped) — fine for a status display.
const RING: usize = 1024;

/// Completion timestamps (ms since boot), newest at the back, pruned to 5 min.
static JOB_LOG: Mutex<RefCell<Deque<u64, RING>>> = Mutex::new(RefCell::new(Deque::new()));

/// Milliseconds since boot.
pub fn now_ms() -> u64 {
    embassy_time::Instant::now().as_millis()
}

/// Record one completed PySpell job (call from each worker on completion).
pub fn record_job() {
    let now = now_ms();
    critical_section::with(|cs| {
        let mut log = JOB_LOG.borrow(cs).borrow_mut();
        // Drop entries older than the widest window.
        while let Some(&front) = log.front() {
            if now.saturating_sub(front) > WIN_60MIN {
                log.pop_front();
            } else {
                break;
            }
        }
        if log.is_full() {
            log.pop_front();
        }
        let _ = log.push_back(now);
    });
}

/// (jobs in last 10 s, 60 s, 5 min, 60 min).
pub fn counts() -> (u32, u32, u32, u32) {
    let now = now_ms();
    critical_section::with(|cs| {
        let log = JOB_LOG.borrow(cs).borrow();
        let (mut c10, mut c60, mut c300, mut c3600) = (0, 0, 0, 0);
        for &t in log.iter() {
            let age = now.saturating_sub(t);
            if age <= WIN_10S {
                c10 += 1;
            }
            if age <= WIN_60S {
                c60 += 1;
            }
            if age <= WIN_5MIN {
                c300 += 1;
            }
            if age <= WIN_60MIN {
                c3600 += 1;
            }
        }
        (c10, c60, c300, c3600)
    })
}

/// Render the counters + free heap onto an embedded-graphics target (160×80).
pub fn draw<D>(display: &mut D, free_heap_kb: u32) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    let (c10, c60, c300, c3600) = counts();

    display.clear(Rgb565::BLACK)?;

    let title = MonoTextStyle::new(&FONT_9X15_BOLD, Rgb565::CSS_ORANGE);
    let label = MonoTextStyle::new(&FONT_6X10, Rgb565::WHITE);
    let value = MonoTextStyle::new(&FONT_6X10, Rgb565::CSS_LIME);

    Text::new("PySpell jobs", Point::new(4, 12), title).draw(display)?;

    let mut line: String<24> = String::new();
    for (i, (lbl, n)) in [
        ("10s ", c10),
        ("60s ", c60),
        ("5min", c300),
        ("60m ", c3600),
    ]
    .iter()
    .enumerate()
    {
        line.clear();
        let _ = write!(line, "{}: {}", lbl, n);
        Text::new(&line, Point::new(4, 26 + i as i32 * 12), value).draw(display)?;
    }

    line.clear();
    let _ = write!(line, "heap {}kB", free_heap_kb);
    Text::new(&line, Point::new(4, 78), label).draw(display)?;

    Ok(())
}
