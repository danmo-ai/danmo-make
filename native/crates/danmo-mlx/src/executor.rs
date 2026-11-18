use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::cache::ModelCache;
use crate::{Device, DeviceType, Error, Result, Stream};

type AnyValue = Box<dyn Any + Send>;
type Job =
    Box<dyn FnOnce(&Device, &Stream, &mut ModelCache, &EventSink) -> Result<AnyValue> + Send>;

enum Command {
    Run(Job, Sender<Result<AnyValue>>),
    Shutdown,
}

/// Plain progress and log records. They cross threads; MLX arrays do not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorEvent {
    Progress { done: u32, total: u32 },
    Log { level: String, message: String },
}

/// Fan-out for events emitted on the executor thread. A disconnected
/// subscriber is dropped. Emission does not wait for the subscriber.
#[derive(Clone)]
pub struct EventSink {
    subscribers: Arc<Mutex<Vec<Sender<ExecutorEvent>>>>,
}

impl EventSink {
    fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn progress(&self, done: u32, total: u32) {
        self.emit(ExecutorEvent::Progress { done, total });
    }

    pub fn log(&self, level: impl Into<String>, message: impl Into<String>) {
        self.emit(ExecutorEvent::Log {
            level: level.into(),
            message: message.into(),
        });
    }

    fn emit(&self, event: ExecutorEvent) {
        let mut subscribers = self
            .subscribers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        subscribers.retain(|sender| sender.send(event.clone()).is_ok());
    }

    fn subscribe(&self) -> Receiver<ExecutorEvent> {
        let (sender, receiver) = channel();
        self.subscribers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(sender);
        receiver
    }
}

pub struct Executor {
    sender: Sender<Command>,
    thread: Option<JoinHandle<()>>,
    events: EventSink,
}

impl Executor {
    /// Spawn the long-lived MLX thread. The resident cache starts with no
    /// byte budget (`usize::MAX`): inserts are recorded and nothing is
    /// evicted until `set_budget`.
    pub fn spawn(device_type: DeviceType) -> Result<Self> {
        let (sender, receiver) = channel::<Command>();
        let (init_sender, init_receiver) = channel::<Result<()>>();
        let events = EventSink::new();
        let thread_events = events.clone();
        let thread = std::thread::Builder::new()
            .name("danmo-mlx-executor".into())
            .spawn(move || {
                let device = match Device::new(device_type, 0) {
                    Ok(device) => device,
                    Err(error) => {
                        let _ = init_sender.send(Err(error));
                        return;
                    }
                };
                let stream = match Stream::new(&device) {
                    Ok(stream) => stream,
                    Err(error) => {
                        let _ = init_sender.send(Err(error));
                        return;
                    }
                };
                if init_sender.send(Ok(())).is_err() {
                    return;
                }
                let mut cache = ModelCache::new(usize::MAX);
                while let Ok(command) = receiver.recv() {
                    match command {
                        Command::Run(job, response) => {
                            let _ =
                                response.send(job(&device, &stream, &mut cache, &thread_events));
                        }
                        Command::Shutdown => break,
                    }
                }
            })
            .map_err(|error| Error::new("Executor::spawn", error.to_string()))?;
        match init_receiver.recv() {
            Ok(Ok(())) => Ok(Self {
                sender,
                thread: Some(thread),
                events,
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(error) => {
                let _ = thread.join();
                Err(Error::new("Executor::spawn", error.to_string()))
            }
        }
    }

    /// Events emitted by jobs. The receiver may live on another thread.
    /// The channel is unbounded so a slow subscriber cannot stall the device thread.
    pub fn subscribe(&self) -> Receiver<ExecutorEvent> {
        self.events.subscribe()
    }

    pub fn set_budget(&self, bytes: usize) -> Result<()> {
        self.execute("ModelCache::set_budget", move |_, _, cache, _| {
            cache.set_budget(bytes);
            Ok(())
        })
    }

    pub fn execute<T, F>(&self, operation: &'static str, function: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&Device, &Stream, &mut ModelCache, &EventSink) -> Result<T> + Send + 'static,
    {
        let (response, receiver) = channel();
        let job: Job = Box::new(move |device, stream, cache, events| {
            let value = catch_unwind(AssertUnwindSafe(|| function(device, stream, cache, events)))
                .map_err(|_| Error::new(operation, "MLX executor job panicked"))??;
            Ok(Box::new(value))
        });
        self.sender
            .send(Command::Run(job, response))
            .map_err(|error| Error::new(operation, error.to_string()))?;
        let value = receiver
            .recv()
            .map_err(|error| Error::new(operation, error.to_string()))??;
        value
            .downcast::<T>()
            .map(|value| *value)
            .map_err(|_| Error::new(operation, "executor returned an unexpected result type"))
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        let _ = self.sender.send(Command::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
