//! Device-side display capability for PySpell `show(x)`.
//!
//! Draws the text on the T-Dongle's ST7735 by posting to the existing UI thread
//! (`ui::send`), reusing the same path the web control panel's "show text" uses.
//! Gated by `config::DISPLAY_ALLOW`; `config::DISPLAY_REVERT_SECONDS` controls
//! whether the screen returns to the default info view after a delay (0 = keep
//! showing the PySpell text until something else changes it).

use std::time::Duration;

use pyspell_core::{Display, DslError};

use crate::ui::{self, UiCommand};

pub struct DeviceDisplay;

impl Display for DeviceDisplay {
    fn show(&self, text: &str) -> Result<(), DslError> {
        if !crate::config::DISPLAY_ALLOW {
            return Err(DslError::Display(String::from("display disabled in config")));
        }
        ui::send(UiCommand::Text(text.to_string()));

        let revert = crate::config::DISPLAY_REVERT_SECONDS;
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
