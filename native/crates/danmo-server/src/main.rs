//! Axum entry. Binary name stays `danqing-api` so the desktop sidecar path is unchanged.

mod app;
mod download;
mod engine_host;
mod llm;
mod product;
mod setup;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port: u16 = match std::env::var("DANQING_HTTP_PORT") {
        Ok(value) => value.parse()?,
        Err(_) => 7800,
    };
    let host = std::env::var("DANQING_HTTP_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let db = std::env::var("DANQING_DB").unwrap_or_else(|_| "db/studio.db".into());
    let registry = std::env::var("DANQING_REGISTRY")
        .unwrap_or_else(|_| "default_config/models_registry.json".into());
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    let app = app::try_router(std::path::Path::new(&db), std::path::Path::new(&registry))
        .map_err(|error| format!("API initialization failed: {error}"))?;
    tracing::info!("danqing-api ggml backend on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
