//! EIS (libei) mouse backend for Wayland.
//!
//! ## Why this exists
//!
//! On Wayland, injecting through uinput makes the compositor resolve absolute
//! coordinates against `seat.active_output()` only — see `cosmic-comp#2103` and
//! the `PointerMotionAbsolute` handler in `cosmic-comp/src/input/mod.rs`. With
//! several monitors shared, the pointer therefore cannot reach a monitor that
//! is not the active one, and the coordinate range is additionally scaled into
//! that single output.
//!
//! The RemoteDesktop portal's `ConnectToEIS` gives us an ei session where
//! absolute coordinates live in the compositor's *global* logical space, so the
//! compositor locates the output itself. cosmic-comp's own comment calls this
//! "the KWin/mutter model".
//!
//! ## Scope
//!
//! Only the MOUSE backend is replaced. The keyboard keeps using uinput:
//!
//! * `get_key_state()` (CapsLock / NumLock synchronisation, used by
//!   `input_service::LockModesHandler`) has no libei equivalent, and
//! * keyboard input does not depend on cursor coordinates, so it never had the
//!   multi-monitor problem.
//!
//! ## Why activation is asynchronous
//!
//! Acquiring the ei session goes through `org.freedesktop.portal.RemoteDesktop`,
//! which needs the user to approve a prompt — and the approval is **not**
//! persisted: the permission store never gains a `remotedesktop` entry, and the
//! `Start` response carries no restore token, so every new server process needs
//! a fresh approval.
//!
//! `--server` is restarted by the service roughly every hour, and while that
//! happens the operator is often remote and physically cannot see or click the
//! prompt. Blocking the startup path on it would (a) stall `start_server` until
//! the machine stops listening at all, or (b) time out and silently leave the
//! operator on uinput with a broken multi-monitor pointer.
//!
//! So activation runs on its own thread and never touches startup:
//!
//! ```text
//! server start ──► uinput mouse installed (works immediately, single output)
//!                        │
//!                        └─ background thread waits for the portal prompt
//!                                │ approved
//!                                ▼
//!                        mouse backend swapped to EIS (multi-monitor aware)
//! ```
//!
//! The swap can happen at any time and does not need a restart.
//!
//! ## Why `enigo` 0.6.x is a second dependency
//!
//! The vendored `libs/enigo` is 0.0.14 and has no libei support; crates.io 0.6.x
//! does (`enigo::linux::libei`). Rewriting the ei protocol and the portal
//! handshake here would be a lot of untested code, so both copies are used: the
//! old one keeps serving the rest of the tree unchanged, the new one only here.
//!
//! ## Why the `Enigo` value lives on its own thread
//!
//! enigo's libei backend owns a tokio `Runtime`. Building it inside a tokio
//! context makes enigo fall back to `futures::executor::block_on` (cannot drive
//! tokio I/O); dropping it inside a tokio context panics —
//! "Cannot drop a runtime in a context where blocking is not allowed". Both the
//! activation thread and the worker thread are plain `std` threads, so neither
//! applies.

use enigo::MouseControllable;
use hbb_common::{anyhow, log};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Duration;

// The 0.6.x copy: it owns the libei backend.
use enigo6::{Axis, Button, Coordinate, Direction, Enigo, Mouse as _, Settings};

/// How long a single activation attempt waits for the portal before giving up
/// and retrying. This happens off the startup path, so it can be generous.
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(300);

/// Delay between failed activation attempts.
const RETRY_INTERVAL: Duration = Duration::from_secs(30);

enum Cmd {
    MoveTo(i32, i32),
    MoveRel(i32, i32),
    Down(enigo::MouseButton),
    Up(enigo::MouseButton),
    Click(enigo::MouseButton),
    ScrollX(i32),
    ScrollY(i32),
}

/// Keeps the worker thread (and therefore its `Enigo`) alive. Dropping this
/// closes the command channel and lets the worker exit.
struct Worker {
    tx: mpsc::Sender<Cmd>,
}

lazy_static::lazy_static! {
    static ref WORKER: Mutex<Option<Worker>> = Mutex::new(None);
}

/// Mouse backend that forwards to the worker thread owning the ei session.
pub struct EisMouse {
    tx: Option<mpsc::Sender<Cmd>>,
}

impl EisMouse {
    fn send(&self, cmd: Cmd) {
        if let Some(tx) = &self.tx {
            // Unbounded and non-blocking: a wedged backend must never stall the
            // input thread.
            let _ = tx.send(cmd);
        }
    }
}

impl Drop for EisMouse {
    fn drop(&mut self) {
        self.tx.take();
    }
}

impl MouseControllable for EisMouse {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn mouse_move_to(&mut self, x: i32, y: i32) {
        // ei absolute coordinates are global logical desktop coordinates.
        // libei rejects negatives, so clamp instead of poisoning the backend
        // with a stale client coordinate.
        self.send(Cmd::MoveTo(x.max(0), y.max(0)));
    }

    fn mouse_move_relative(&mut self, x: i32, y: i32) {
        self.send(Cmd::MoveRel(x, y));
    }

    fn mouse_down(&mut self, button: enigo::MouseButton) -> enigo::ResultType {
        self.send(Cmd::Down(button));
        Ok(())
    }

    fn mouse_up(&mut self, button: enigo::MouseButton) {
        self.send(Cmd::Up(button));
    }

    fn mouse_click(&mut self, button: enigo::MouseButton) {
        self.send(Cmd::Click(button));
    }

    fn mouse_scroll_x(&mut self, length: i32) {
        self.send(Cmd::ScrollX(length));
    }

    fn mouse_scroll_y(&mut self, length: i32) {
        self.send(Cmd::ScrollY(length));
    }
}

fn map_button(button: enigo::MouseButton) -> Option<Button> {
    Some(match button {
        enigo::MouseButton::Left => Button::Left,
        enigo::MouseButton::Middle => Button::Middle,
        enigo::MouseButton::Right => Button::Right,
        enigo::MouseButton::Back => Button::Back,
        enigo::MouseButton::Forward => Button::Forward,
        // Scroll "buttons" are expressed through `scroll()` on this backend.
        _ => return None,
    })
}

/// Start the asynchronous activation thread. Returns immediately; uinput keeps
/// serving input until (and unless) the ei session comes up.
pub fn spawn_background_activation() {
    let spawned = std::thread::Builder::new()
        .name("eis-activate".into())
        .spawn(move || {
            // The approval prompt may sit unanswered for a long time; keep
            // retrying quietly rather than giving up on the session.
            let mut attempt = 0u32;
            loop {
                attempt += 1;
                log::info!(
                    "[EIS] activation attempt #{attempt}: waiting for the RemoteDesktop portal \
                     (uinput stays active meanwhile)"
                );
                match start_worker() {
                    Ok((tx, worker)) => {
                        // Swap the backend under the same lock the input path uses,
                        // so no event can be lost between the two backends.
                        let old = super::input_service::swap_mouse_backend(Box::new(EisMouse {
                            tx: Some(tx),
                        }));
                        // Keep the worker (and its Enigo) alive for the process
                        // lifetime. Dropping the previous one closes its channel.
                        *WORKER.lock().unwrap() = Some(worker);
                        if let Some(old) = old {
                            // `UInputMouse` owns a tokio Runtime; this thread is a
                            // plain std thread, which is a safe place to drop it.
                            drop(old);
                        }
                        log::info!("[EIS] mouse backend switched to EIS (multi-monitor aware)");
                        return;
                    }
                    Err(e) => {
                        log::warn!(
                            "[EIS] activation attempt #{attempt} failed: {e}; retrying in {}s",
                            RETRY_INTERVAL.as_secs()
                        );
                        std::thread::sleep(RETRY_INTERVAL);
                    }
                }
            }
        });
    if let Err(e) = spawned {
        log::warn!("[EIS] failed to spawn the activation thread: {e}; staying on uinput");
    }
}

/// Bring up one ei session on a dedicated thread and wait (bounded) for it.
fn start_worker() -> hbb_common::ResultType<(mpsc::Sender<Cmd>, Worker)> {
    let (tx, rx) = mpsc::channel::<Cmd>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("eis-mouse".into())
        .spawn(move || {
            log::info!("[EIS] connecting to the RemoteDesktop portal...");
            let mut enigo = match Enigo::new(&Settings::default()) {
                Ok(enigo) => {
                    log::info!("[EIS] portal session established");
                    let _ = ready_tx.send(Ok(()));
                    enigo
                }
                Err(e) => {
                    log::warn!("[EIS] portal session failed: {e}");
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };

            // Drain commands until the backend is dropped. `enigo` is created and
            // destroyed on this thread, never inside tokio.
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    Cmd::MoveTo(x, y) => {
                        if let Err(e) = enigo.move_mouse(x, y, Coordinate::Abs) {
                            log::debug!("[EIS] move abs ({x},{y}) failed: {e:?}");
                        }
                    }
                    Cmd::MoveRel(x, y) => {
                        if let Err(e) = enigo.move_mouse(x, y, Coordinate::Rel) {
                            log::debug!("[EIS] move rel ({x},{y}) failed: {e:?}");
                        }
                    }
                    Cmd::Down(b) => {
                        if let Some(b) = map_button(b) {
                            let _ = enigo.button(b, Direction::Press);
                        }
                    }
                    Cmd::Up(b) => {
                        if let Some(b) = map_button(b) {
                            let _ = enigo.button(b, Direction::Release);
                        }
                    }
                    Cmd::Click(b) => {
                        if let Some(b) = map_button(b) {
                            let _ = enigo.button(b, Direction::Click);
                        }
                    }
                    Cmd::ScrollX(n) => {
                        if let Err(e) = enigo.scroll(n, Axis::Horizontal) {
                            log::debug!("[EIS] scroll x {n} failed: {e:?}");
                        }
                    }
                    Cmd::ScrollY(n) => {
                        if let Err(e) = enigo.scroll(n, Axis::Vertical) {
                            log::debug!("[EIS] scroll y {n} failed: {e:?}");
                        }
                    }
                }
            }
            log::info!("[EIS] worker stopped");
            // `enigo` drops here, on this plain thread — the safe place.
        })
        .map_err(|e| anyhow::anyhow!("failed to spawn eis worker thread: {e}"))?;

    match ready_rx.recv_timeout(ATTEMPT_TIMEOUT) {
        Ok(Ok(())) => Ok((tx.clone(), Worker { tx })),
        Ok(Err(e)) => Err(anyhow::anyhow!("portal refused: {e}")),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow::anyhow!(
            "portal did not answer within {}s",
            ATTEMPT_TIMEOUT.as_secs()
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err(anyhow::anyhow!("worker thread died during handshake"))
        }
    }
}
