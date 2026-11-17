use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Sender};
use std::thread::JoinHandle;

use crate::{Device, DeviceType, Error, Result, Stream};

type AnyValue = Box<dyn Any + Send>;
type Job = Box<dyn FnOnce(&Device, &Stream) -> Result<AnyValue> + Send>;

enum Command {
    Run(Job, Sender<Result<AnyValue>>),
    Shutdown,
}

pub struct Executor {
    sender: Sender<Command>,
    thread: Option<JoinHandle<()>>,
}

impl Executor {
    pub fn spawn(device_type: DeviceType) -> Result<Self> {
        let (sender, receiver) = channel::<Command>();
        let (init_sender, init_receiver) = channel::<Result<()>>();
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
                while let Ok(command) = receiver.recv() {
                    match command {
                        Command::Run(job, response) => {
                            let _ = response.send(job(&device, &stream));
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

    pub fn execute<T, F>(&self, operation: &'static str, function: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&Device, &Stream) -> Result<T> + Send + 'static,
    {
        let (response, receiver) = channel();
        let job: Job = Box::new(move |device, stream| {
            let value = catch_unwind(AssertUnwindSafe(|| function(device, stream)))
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
