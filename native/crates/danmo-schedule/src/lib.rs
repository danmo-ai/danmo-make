//! Single worker queue. Image tasks dequeue first when `queue_image_first` is set.

use danmo_store::{AssetRow, CanvasRow, LogRow, StoreError, TaskRow, TaskStore};
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

    fn lock(&self) -> std::sync::MutexGuard<'_, TaskStore> {
        self.store.lock().expect("task store")
    }

    pub fn submit(
        &self,
        id: &str,
        kind: &str,
        model_id: &str,
        params: &str,
        priority: i64,
    ) -> Result<(), StoreError> {
        self.lock()
            .insert_task(id, kind, model_id, params, priority)
    }

    pub fn dequeue(&self) -> Result<Option<TaskRow>, StoreError> {
        let store = self.lock();
        let queued = store.list_queued(self.queue_image_first)?;
        for next in queued {
            if store.mark_running(&next.id)? == 1 {
                return Ok(Some(next));
            }
        }
        Ok(None)
    }

    pub fn finish_ok(&self, id: &str, result: &str) -> Result<(), StoreError> {
        self.lock()
            .finish(id, "completed", 1.0, Some(result), None)?;
        Ok(())
    }

    pub fn finish_err(&self, id: &str, message: &str) -> Result<(), StoreError> {
        self.lock().finish(id, "failed", 1.0, None, Some(message))?;
        Ok(())
    }

    pub fn cancel(&self, id: &str) -> Result<&'static str, StoreError> {
        let store = self.lock();
        let Some(row) = store.get(id)? else {
            return Ok("missing");
        };
        if row.status != "queued" && row.status != "running" {
            return Ok("not_cancellable");
        }
        store.finish(id, "cancelled", row.progress, None, Some("cancelled"))?;
        Ok("ok")
    }

    pub fn set_priority(&self, id: &str, priority: i64) -> Result<&'static str, StoreError> {
        let store = self.lock();
        let Some(row) = store.get(id)? else {
            return Ok("missing");
        };
        if row.status != "queued" {
            return Ok("not_queued");
        }
        if store.set_priority(id, priority)? == 0 {
            return Ok("not_queued");
        }
        Ok("ok")
    }

    pub fn progress(&self, id: &str, progress: f64, message: &str) -> Result<(), StoreError> {
        let store = self.lock();
        store.set_progress(id, progress)?;
        store.log(id, "info", message)
    }

    pub fn get(&self, id: &str) -> Result<Option<TaskRow>, StoreError> {
        self.lock().get(id)
    }

    pub fn list_tasks(
        &self,
        limit: i64,
        offset: i64,
        kind: Option<&str>,
        status: Option<&str>,
        since: Option<&str>,
    ) -> Result<Vec<TaskRow>, StoreError> {
        self.lock().list_tasks(limit, offset, kind, status, since)
    }

    pub fn logs(&self, id: &str, offset: i64, limit: i64) -> Result<Vec<LogRow>, StoreError> {
        self.lock().logs(id, offset, limit)
    }

    pub fn insert_asset(&self, row: &AssetRow) -> Result<(), StoreError> {
        self.lock().insert_asset(row)
    }

    pub fn get_asset(&self, id: &str) -> Result<Option<AssetRow>, StoreError> {
        self.lock().get_asset(id)
    }

    pub fn list_assets(
        &self,
        kind: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<(Vec<AssetRow>, i64), StoreError> {
        self.lock().list_assets(kind, limit, offset)
    }

    pub fn delete_asset(&self, id: &str) -> Result<Option<AssetRow>, StoreError> {
        self.lock().delete_asset(id)
    }

    pub fn child_assets(&self, parent_id: &str) -> Result<Vec<AssetRow>, StoreError> {
        self.lock().child_assets(parent_id)
    }

    pub fn insert_canvas(&self, row: &CanvasRow) -> Result<(), StoreError> {
        self.lock().insert_canvas(row)
    }

    pub fn update_canvas(
        &self,
        id: &str,
        title: Option<&str>,
        state_json: Option<&str>,
        updated_at: &str,
    ) -> Result<bool, StoreError> {
        self.lock().update_canvas(id, title, state_json, updated_at)
    }

    pub fn get_canvas(&self, id: &str) -> Result<Option<CanvasRow>, StoreError> {
        self.lock().get_canvas(id)
    }

    pub fn list_canvas(&self, media: &str, limit: i64) -> Result<Vec<CanvasRow>, StoreError> {
        self.lock().list_canvas(media, limit)
    }

    pub fn delete_canvas(&self, id: &str) -> Result<bool, StoreError> {
        self.lock().delete_canvas(id)
    }
}
