//! Single worker queue. Image tasks dequeue first when `queue_image_first` is set.

use danmo_contracts::CancellationToken;
use danmo_store::{AssetRow, CanvasRow, LogRow, StoreError, TaskRow, TaskStore};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Live step detail for a running task. Kept in memory (not `task_logs`) so
/// SSE / queue UI can show denoise `N/M` while OTel only exports finished
/// phase spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressTip {
    pub phase: String,
    pub step: u32,
    pub total: u32,
}

#[derive(Clone)]
pub struct Scheduler {
    store: Arc<Mutex<TaskStore>>,
    cancellations: Arc<Mutex<HashMap<String, CancellationToken>>>,
    progress_tips: Arc<Mutex<HashMap<String, ProgressTip>>>,
    pub queue_image_first: bool,
}

impl Scheduler {
    pub fn open(db: &Path, queue_image_first: bool) -> Result<Self, StoreError> {
        let store = TaskStore::open(db)?;
        let interrupted = store.fail_interrupted_tasks()?;
        if interrupted > 0 {
            eprintln!("recovered {interrupted} task(s) interrupted by the previous process");
        }
        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            cancellations: Arc::new(Mutex::new(HashMap::new())),
            progress_tips: Arc::new(Mutex::new(HashMap::new())),
            queue_image_first,
        })
    }

    pub fn with_store(store: TaskStore, queue_image_first: bool) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            cancellations: Arc::new(Mutex::new(HashMap::new())),
            progress_tips: Arc::new(Mutex::new(HashMap::new())),
            queue_image_first,
        }
    }

    pub fn start_cancellable(&self, id: &str) -> CancellationToken {
        let token = CancellationToken::default();
        self.cancellations
            .lock()
            .expect("cancellation registry")
            .insert(id.into(), token.clone());
        token
    }

    pub fn finish_cancellable(&self, id: &str) {
        self.cancellations
            .lock()
            .expect("cancellation registry")
            .remove(id);
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

    /// Submit a task directly in the running state, bypassing the queue
    /// worker. For jobs that execute on their own thread (LoRA training).
    pub fn submit_running(
        &self,
        id: &str,
        kind: &str,
        model_id: &str,
        params: &str,
        priority: i64,
    ) -> Result<(), StoreError> {
        self.lock()
            .insert_running_task(id, kind, model_id, params, priority)
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
        self.clear_progress_tip(id);
        self.lock()
            .finish(id, "completed", 1.0, Some(result), None)?;
        Ok(())
    }

    pub fn finish_err(&self, id: &str, message: &str) -> Result<(), StoreError> {
        self.clear_progress_tip(id);
        self.lock().finish(id, "failed", 1.0, None, Some(message))?;
        Ok(())
    }

    pub fn finish_cancelled(&self, id: &str) -> Result<(), StoreError> {
        self.clear_progress_tip(id);
        self.lock()
            .finish(id, "cancelled", 1.0, None, Some("cancelled"))?;
        Ok(())
    }

    pub fn cancel(&self, id: &str) -> Result<&'static str, StoreError> {
        if let Some(token) = self
            .cancellations
            .lock()
            .expect("cancellation registry")
            .get(id)
            .cloned()
        {
            token.cancel();
        }
        let store = self.lock();
        let Some(row) = store.get(id)? else {
            return Ok("missing");
        };
        if row.status != "queued" && row.status != "running" {
            return Ok("not_cancellable");
        }
        store.finish(id, "cancelled", row.progress, None, Some("cancelled"))?;
        drop(store);
        self.clear_progress_tip(id);
        Ok("ok")
    }

    /// Permanently remove a finished task: its row, its logs, and (via the
    /// caller) its trace spans. Running and queued tasks must be cancelled
    /// first so a live worker never writes into a deleted row.
    pub fn purge(&self, id: &str) -> Result<&'static str, StoreError> {
        let result = self.lock().purge_task(id)?;
        self.clear_progress_tip(id);
        Ok(result)
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

    /// Append one line to a task's log without moving its progress bar.
    pub fn log(&self, id: &str, level: &str, message: &str) -> Result<(), StoreError> {
        self.lock().log(id, level, message)
    }

    /// Move the progress bar only; no log row.
    pub fn set_progress(&self, id: &str, progress: f64) -> Result<(), StoreError> {
        self.lock().set_progress(id, progress)
    }

    /// Update `tasks.progress` and the in-memory step tip used by SSE / graph.
    /// Does not insert a `task_logs` row — generation progress lives in OTel.
    pub fn set_progress_tip(
        &self,
        id: &str,
        progress: f64,
        phase: &str,
        step: u32,
        total: u32,
    ) -> Result<(), StoreError> {
        self.lock().set_progress(id, progress)?;
        self.progress_tips.lock().expect("progress tips").insert(
            id.into(),
            ProgressTip {
                phase: phase.into(),
                step,
                total,
            },
        );
        Ok(())
    }

    pub fn progress_tip(&self, id: &str) -> Option<ProgressTip> {
        self.progress_tips
            .lock()
            .expect("progress tips")
            .get(id)
            .cloned()
    }

    pub fn clear_progress_tip(&self, id: &str) {
        self.progress_tips.lock().expect("progress tips").remove(id);
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

    pub fn recent_logs(&self, id: &str, limit: i64) -> Result<Vec<LogRow>, StoreError> {
        self.lock().recent_logs(id, limit)
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
        exclude_upload_refs: bool,
    ) -> Result<(Vec<AssetRow>, i64), StoreError> {
        self.lock()
            .list_assets(kind, limit, offset, exclude_upload_refs)
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

    pub fn update_asset_metadata(&self, id: &str, metadata: &str) -> Result<bool, StoreError> {
        self.lock().update_asset_metadata(id, metadata)
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
