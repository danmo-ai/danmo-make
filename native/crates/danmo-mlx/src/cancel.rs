use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Cross-thread cancellation flag for long-running inference.
///
/// Only plain data crosses threads (plan §1.3): the control plane holds a
/// clone and calls `cancel()`, while the executor thread checks
/// `is_cancelled()` at step/token/tile boundaries. MLX arrays themselves stay
/// thread-affine and are never shared.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::CancelToken;

    #[test]
    fn cancel_crosses_threads_and_stays_cancelled() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        let control = token.clone();
        std::thread::spawn(move || control.cancel())
            .join()
            .unwrap();
        assert!(token.is_cancelled());
        assert!(token.clone().is_cancelled());
    }
}
