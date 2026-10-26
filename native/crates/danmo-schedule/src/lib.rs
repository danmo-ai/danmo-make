//! Single worker queue. Image tasks dequeue first when `queue_image_first` is set.

use danmo_store::{StoreError, TaskRow, TaskStore};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Scheduler {
    store: Arc<Mutex<TaskStore>>,
    pub queue_image_first: bool,
}

impl Scheduler {
    pub fn open(db: &Path, queue_image_first: bool) -> Result<Self, StoreError> {
        Ok(Self {
            store: Arc::new(Mutex::new(TaskStore::open(db)?)),
            queue_image_first,
        })
    }

    pub fn with_store(store: TaskStore, queue_image_first: bool) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            queue_image_first,
        }
    }

    pub fn submit(
        &self,
        id: &str,
        kind: &str,
        model_id: &str,
        params: &str,
        priority: i64,
    ) -> Result<(), StoreError> {
        self.store
            .lock()
            .expect("task store")
            .insert_task(id, kind, model_id, params, priority)
    }

    pub fn dequeue(&self) -> Result<Option<TaskRow>, StoreError> {
        let store = self.store.lock().expect("task store");
        let mut queued = store.list_queued(self.queue_image_first)?;
        let Some(next) = queued.drain(..).next() else {
            return Ok(None);
        };
        store.update_status(&next.id, "running", 0.0, None, None)?;
        Ok(Some(next))
    }

    pub fn finish_ok(&self, id: &str, result: &str) -> Result<(), StoreError> {
        self.store.lock().expect("task store").update_status(
            id,
            "succeeded",
            1.0,
            Some(result),
            None,
        )
    }

    pub fn finish_err(&self, id: &str, message: &str) -> Result<(), StoreError> {
        self.store
            .lock()
            .expect("task store")
            .update_status(id, "failed", 1.0, None, Some(message))
    }

    pub fn progress(&self, id: &str, progress: f64, message: &str) -> Result<(), StoreError> {
        let store = self.store.lock().expect("task store");
        store.update_status(id, "running", progress, None, None)?;
        store.log(id, "info", message)
    }

    pub fn get(&self, id: &str) -> Result<Option<TaskRow>, StoreError> {
        self.store.lock().expect("task store").get(id)
    }
}
