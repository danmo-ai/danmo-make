use super::{BackendKind, GgmlContext};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone)]
pub struct CapacityReport {
    pub backend: BackendKind,
    pub weight_bytes: usize,
    pub conditioning_bytes: usize,
    pub graph_bytes: usize,
    pub graphs: usize,
    pub available_bytes: usize,
    pub device_bytes: usize,
}

impl CapacityReport {
    pub fn managed_bytes(&self) -> usize {
        self.weight_bytes + self.conditioning_bytes + self.graph_bytes
    }

    pub fn summary(&self) -> String {
        let gib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        format!(
            "capacity backend={} weights={:.2}GiB conditioning={:.2}GiB graphs={:.2}GiB/{} managed={:.2}GiB available={:.2}GiB total={:.2}GiB",
            self.backend.name(),
            gib(self.weight_bytes),
            gib(self.conditioning_bytes),
            gib(self.graph_bytes),
            self.graphs,
            gib(self.managed_bytes()),
            gib(self.available_bytes),
            gib(self.device_bytes),
        )
    }
}

fn graphs() -> &'static Mutex<HashMap<usize, (BackendKind, usize)>> {
    static GRAPHS: OnceLock<Mutex<HashMap<usize, (BackendKind, usize)>>> = OnceLock::new();
    GRAPHS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn register(id: usize, backend: BackendKind, bytes: usize) {
    graphs()
        .lock()
        .expect("capacity registry")
        .insert(id, (backend, bytes));
}

pub(crate) fn unregister(id: usize) {
    graphs().lock().expect("capacity registry").remove(&id);
}

pub fn snapshot(ctx: &GgmlContext) -> CapacityReport {
    let graphs = graphs().lock().expect("capacity registry");
    let mut graph_bytes = 0usize;
    let mut count = 0usize;
    for (backend, bytes) in graphs.values() {
        if *backend == ctx.kind() {
            graph_bytes = graph_bytes.saturating_add(*bytes);
            count += 1;
        }
    }
    let (available_bytes, device_bytes) = ctx.memory();
    CapacityReport {
        backend: ctx.kind(),
        weight_bytes: crate::weights::resident_bytes(),
        conditioning_bytes: crate::common::conditioning::resident_bytes(),
        graph_bytes,
        graphs: count,
        available_bytes,
        device_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_registration_changes_the_capacity_report() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        register(usize::MAX, BackendKind::Cpu, 4096);
        let during = snapshot(&ctx);
        // Parallel tests register and drop their own graphs, so only the
        // contribution of this registration is a stable lower bound.
        assert!(during.graph_bytes >= 4096);
        assert!(during.summary().contains("managed"));
        unregister(usize::MAX);
        assert!(!graphs().lock().unwrap().contains_key(&usize::MAX));
    }
}
