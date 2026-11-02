//! The embedding model for `understudy serve --understudent auto`: used when it is there, fetched in the background
//! when it is not, and never a reason for the server to fail at startup.

use crate::encoder::{Eg2, Encoder};
use std::sync::{Arc, Mutex};

enum State {
    /// Nothing tried yet.
    Idle,
    /// The download is running in the background.
    Fetching,
    Ready(Arc<dyn Encoder>),
    /// Not available, and why (the first line of the error).
    Failed(String),
}

pub struct LazyEncoder {
    state: Mutex<State>,
    cv: std::sync::Condvar,
    download: bool,
    /// How a model is loaded and fetched (the real one, or a test's).
    load: Box<dyn Fn() -> anyhow::Result<Arc<dyn Encoder>> + Send + Sync>,
    fetch: Box<dyn Fn() -> anyhow::Result<()> + Send + Sync>,
}

impl LazyEncoder {
    /// The EmbeddingGemma 2 checkpoint; `download` lets it fetch the checkpoint (checksum-verified) when it is missing.
    pub fn eg2(download: bool) -> Arc<Self> {
        Self::with(download, Box::new(|| Ok(Arc::new(Eg2::load_default()?) as Arc<dyn Encoder>)), Box::new(|| crate::modelfetch::fetch(None, None).map(|_| ())))
    }

    pub fn with(download: bool, load: Box<dyn Fn() -> anyhow::Result<Arc<dyn Encoder>> + Send + Sync>, fetch: Box<dyn Fn() -> anyhow::Result<()> + Send + Sync>) -> Arc<Self> {
        Arc::new(LazyEncoder { state: Mutex::new(State::Idle), cv: std::sync::Condvar::new(), download, load, fetch })
    }

    /// First use: load the model if it is on disk (a moment), else start fetching it in the background (one log line when
    /// it starts and one when it finishes). Returns at once in the fetching case.
    pub fn start(self: &Arc<Self>) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !matches!(*g, State::Idle) {
            return;
        }
        match (self.load)() {
            Ok(e) => {
                eprintln!("understudy: the embedding model is loaded; the encoder designs join the race");
                *g = State::Ready(e);
            }
            Err(e) if !self.download => {
                eprintln!("understudy: warning: no embedding model ({}); racing the model-free designs only (--no-model-download is set; `understudy model fetch` gets it)", first_line(&e));
                *g = State::Failed(first_line(&e));
            }
            Err(_) => {
                eprintln!("understudy: fetching the embedding model (EmbeddingGemma 2, about 1.5 GB, checksum-verified) in the background; the model-free designs run meanwhile");
                *g = State::Fetching;
                let me = self.clone();
                std::thread::Builder::new()
                    .name("understudy-model-fetch".into())
                    .spawn(move || {
                        let r = (me.fetch)().and_then(|()| (me.load)());
                        let mut g = me.state.lock().unwrap_or_else(|e| e.into_inner());
                        *g = match r {
                            Ok(e) => {
                                eprintln!("understudy: the embedding model is ready; the encoder designs join the race");
                                State::Ready(e)
                            }
                            Err(e) => {
                                eprintln!("understudy: warning: the embedding model download failed ({}); continuing with the model-free designs (retry with `understudy model fetch`)", first_line(&e));
                                State::Failed(first_line(&e))
                            }
                        };
                        me.cv.notify_all();
                    })
                    .ok();
            }
        }
    }

    /// The encoder, if it is ready now.
    pub fn ready(&self) -> Option<Arc<dyn Encoder>> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            State::Ready(e) => Some(e.clone()),
            _ => None,
        }
    }

    /// The encoder, waiting for a fetch in progress; an error that names `understudy model fetch` if there is none.
    pub fn wait(self: &Arc<Self>) -> anyhow::Result<Arc<dyn Encoder>> {
        self.start();
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match &*g {
                State::Ready(e) => return Ok(e.clone()),
                State::Failed(m) => anyhow::bail!("this snapshot needs the embedding model, which is not available ({m}): run `understudy model fetch`"),
                _ => g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner()),
            }
        }
    }
}

fn first_line(e: &anyhow::Error) -> String {
    format!("{e:#}").lines().next().unwrap_or("").to_string()
}
