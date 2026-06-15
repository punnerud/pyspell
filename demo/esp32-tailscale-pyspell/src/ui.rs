//! UI control: the command enum driven by the in-tunnel web panel + map-stream
//! status updates, plus a process-global sender so the data-plane and DERP
//! threads can post commands without threading a channel through every call.
//!
//! The single consumer is the `ui` thread spawned in `main.rs`, which owns the
//! display, the backlight pin and the APA102 LED pins.

use std::sync::mpsc::Sender;
use std::sync::OnceLock;

/// One command for the display/LED UI thread.
pub enum UiCommand {
    /// Turn the underside RGB LED on (white, full brightness).
    LedOn,
    /// Turn the underside RGB LED off.
    LedOff,
    /// Backlight on (screen visible).
    BacklightOn,
    /// Backlight off (screen dark, power saving).
    BacklightOff,
    /// Flash the whole screen white/black 3× with 0.5 s between, then restore.
    Flash,
    /// Show arbitrary text on the screen.
    Text(String),
    /// Return to the default Tailscale-info screen.
    DefaultScreen,
    /// Update + redraw the default info screen (from the map stream).
    Status { ts: String, lan: String, online: bool },
}

static UI_TX: OnceLock<Sender<UiCommand>> = OnceLock::new();

/// Register the UI thread's receiver end. Call once, before any sender.
pub fn set_sender(tx: Sender<UiCommand>) {
    let _ = UI_TX.set(tx);
}

/// Post a command to the UI thread (no-op if the thread isn't up yet).
pub fn send(cmd: UiCommand) {
    if let Some(tx) = UI_TX.get() {
        let _ = tx.send(cmd);
    }
}

/// Map a web-panel TCP action to a UI command and dispatch it.
#[cfg(feature = "http-server")]
pub fn dispatch_tcp(a: tailscale_core::tcp::TcpAction) {
    use tailscale_core::tcp::TcpAction;
    let cmd = match a {
        TcpAction::LedOn => UiCommand::LedOn,
        TcpAction::LedOff => UiCommand::LedOff,
        TcpAction::ScreenOn => UiCommand::BacklightOn,
        TcpAction::ScreenOff => UiCommand::BacklightOff,
        TcpAction::Flash => UiCommand::Flash,
        TcpAction::Default => UiCommand::DefaultScreen,
        TcpAction::Text(s) => UiCommand::Text(s),
    };
    send(cmd);
}
