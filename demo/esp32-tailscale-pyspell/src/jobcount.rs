//! Rolling PySpell job counters for the status display: how many jobs completed in
//! the last 10 s / 60 s / 10 min / 60 min. Each completed job (in-tunnel server or
//! LAN worker pool) records its timestamp; `counts()` tallies those within each
//! window. Bounded (CAP entries) so a burst can't grow it without limit.

use std::collections::VecDeque;
use std::sync::Mutex;

use esp_idf_svc::sys::esp_timer_get_time;

static JOBS: Mutex<VecDeque<i64>> = Mutex::new(VecDeque::new());
const CAP: usize = 1024;
const WIN_60MIN_US: i64 = 3_600_000_000;

/// Record one completed PySpell job (any path).
pub fn record() {
    let now = unsafe { esp_timer_get_time() };
    if let Ok(mut q) = JOBS.lock() {
        q.push_back(now);
        let cutoff = now - WIN_60MIN_US;
        while q.front().map_or(false, |&t| t < cutoff) {
            q.pop_front();
        }
        while q.len() > CAP {
            q.pop_front();
        }
    }
}

/// Job counts for (last 10 s, 60 s, 10 min, 60 min).
pub fn counts() -> (u32, u32, u32, u32) {
    let now = unsafe { esp_timer_get_time() };
    let (mut s10, mut s60, mut m10, mut m60) = (0u32, 0u32, 0u32, 0u32);
    if let Ok(q) = JOBS.lock() {
        for &t in q.iter() {
            let age = now - t;
            if age <= 10_000_000 {
                s10 += 1;
            }
            if age <= 60_000_000 {
                s60 += 1;
            }
            if age <= 600_000_000 {
                m10 += 1;
            }
            if age <= WIN_60MIN_US {
                m60 += 1;
            }
        }
    }
    (s10, s60, m10, m60)
}
