//! Custom `getrandom` backends backed by the esp-hal hardware RNG.
//!
//! tailscale-core's crypto pulls two getrandom majors — 0.2 (directly) and 0.3
//! (via `snow`) — and neither has a built-in backend on bare-metal xtensa. We
//! register both against the esp-hal TRNG so the Noise/x25519/etc. crypto works.

use esp_hal::rng::Rng;

/// getrandom 0.2 custom source.
fn esp_getrandom_02(buf: &mut [u8]) -> Result<(), getrandom02::Error> {
    Rng::new().read(buf);
    Ok(())
}
getrandom02::register_custom_getrandom!(esp_getrandom_02);

/// getrandom 0.3 custom backend (selected via `--cfg getrandom_backend="custom"`).
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom03::Error> {
    let slice = unsafe { core::slice::from_raw_parts_mut(dest, len) };
    Rng::new().read(slice);
    Ok(())
}
