//! Axum entry. Binary name stays `danqing-api` so the desktop sidecar path is unchanged.

mod app;
mod download;
mod product;

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("DANQING_HTTP_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7800);
    let host = std::env::var("DANQING_HTTP_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let db = std::env::var("DANQING_DB").unwrap_or_else(|_| "db/studio.db".into());
    let registry = std::env::var("DANQING_REGISTRY")
        .unwrap_or_else(|_| "default_config/models_registry.json".into());
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}"))
        .await
        .unwrap_or_else(|err| panic!("bind {host}:{port}: {err}"));
    let app = app::router(std::path::Path::new(&db), std::path::Path::new(&registry));
    tracing::info!("danqing-api ggml backend on {host}:{port}");
    axum::serve(listener, app).await.expect("serve");
}
