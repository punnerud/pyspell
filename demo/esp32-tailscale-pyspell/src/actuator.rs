//! Device-side actuator capability for PySpell `led(x)` / `flash()`.
//!
//! Drives the T-Dongle's underside APA102 RGB LED by posting to the UI thread
//! (`ui::send`), the same path the web panel's LED buttons use. Kept out of the
//! pure evaluator: pyspell-core only knows the [`Actuator`] trait.

use pyspell_core::{Actuator, DslError};

use crate::ui::{self, UiCommand};

pub struct DeviceActuator;

impl Actuator for DeviceActuator {
    fn led(&self, on: bool, color: Option<(u8, u8, u8)>) -> Result<(), DslError> {
        let cmd = if !on {
            UiCommand::LedOff
        } else {
            match color {
                Some((r, g, b)) => UiCommand::LedColor { r, g, b },
                None => UiCommand::LedOn, // white, full brightness
            }
        };
        ui::send(cmd);
        Ok(())
    }

    fn flash(&self) -> Result<(), DslError> {
        ui::send(UiCommand::LedFlash);
        Ok(())
    }
}
