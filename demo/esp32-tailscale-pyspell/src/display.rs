//! Device-side display capability for PySpell `show(x)`.
//!
//! Draws the text on the T-Dongle's ST7735 by posting to the existing UI thread
//! (`ui::send`), reusing the same path the web control panel's "show text" uses.
//! Gated by `config::DISPLAY_ALLOW`; `config::DISPLAY_REVERT_SECONDS` controls
//! whether the screen returns to the default info view after a delay (0 = keep
//! showing the PySpell text until something else changes it).

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use pyspell_core::{Display, DslError};

use crate::ui::{self, UiCommand};

/// Runtime override for the screen-revert delay (seconds), settable from the web
/// panel's ⚙ so the user can pick how long `show(...)` text stays up before the
/// default screen returns. `u32::MAX` = use the compile-time config default.
static REVERT_OVERRIDE: AtomicU32 = AtomicU32::new(u32::MAX);

/// Set the revert delay (0 = keep showing until something else changes it).
pub fn set_revert_seconds(secs: u32) {
    REVERT_OVERRIDE.store(secs, Ordering::Relaxed);
}

fn revert_seconds() -> u32 {
    match REVERT_OVERRIDE.load(Ordering::Relaxed) {
        u32::MAX => crate::config::DISPLAY_REVERT_SECONDS,
        v => v,
    }
}

pub struct DeviceDisplay;

impl Display for DeviceDisplay {
    fn show(&self, text: &str) -> Result<(), DslError> {
        if !crate::config::DISPLAY_ALLOW {
            return Err(DslError::Display(String::from("display disabled in config")));
        }
        ui::send(UiCommand::Text(text.to_string()));

        let revert = revert_seconds();
        if revert > 0 {
            // A tiny detached timer thread returns the screen to the default view.
            let _ = std::thread::Builder::new().stack_size(3072).spawn(move || {
                std::thread::sleep(Duration::from_secs(revert as u64));
                ui::send(UiCommand::DefaultScreen);
            });
        }
        Ok(())
    }
}
