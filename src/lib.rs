//! A declarative, DataFrame-native chart library for [Polars](https://pola.rs).
//!
//! charcoal turns a Polars [`polars::frame::DataFrame`] into a publication-quality SVG,
//! standalone HTML, or PNG with a single builder chain — no browser, no Python, no C FFI.
//!
//! # Feature Flags
//!
//! | Feature | Enables | Extra dependency |
//! |---------|---------|-----------------|
//! | *(default)* | SVG and HTML output | — |
//! | `static` | [`Chart::save_png`] via pure-Rust `resvg` | `resvg` |
//! | `notebook` | [`Chart::display`] inline in evcxr Jupyter | `evcxr_runtime` |
//!
//! # Quickstart
//!
//! ```rust,no_run
//! use charcoal::{Chart, Theme};
//! use polars::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let df = CsvReader::from_path("iris.csv")?.finish()?;
//!
//!     let chart = Chart::scatter(&df)
//!         .x("sepal_length")
//!         .y("sepal_width")
//!         .color_by("species")
//!         .title("Iris Dataset")
//!         .theme(Theme::Default)
//!         .build()?;
//!
//!     chart.save_html("iris.html")?;
//!     Ok(())
//! }
//! ```
//!
//! # Examples
//!
//! Runnable examples for every chart type live in the `examples/` directory:
//!
//! ```text
//! cargo run --example scatter_iris
//! cargo run --example line_timeseries
//! cargo run --example bar_categorical
//! cargo run --example histogram_distribution
//! cargo run --example heatmap_correlation
//! cargo run --example box_plot_groups
//! cargo run --example area_stacked
//! ```
//!
//! # Notebook Usage
//!
//! To display charts inline in an [evcxr](https://github.com/evcxr/evcxr) Jupyter
//! notebook, enable the `notebook` feature:
//!
//! ```toml
//! [dependencies]
//! charcoal = { version = "0.1", features = ["notebook"] }
//! ```
//!
//! In your first notebook cell, load the dependency:
//!
//! ```text
//! :dep charcoal = { version = "0.1", features = ["notebook"] }
//! use charcoal::Chart;
//! ```
//!
//! Then call `.display()` as the last expression in any subsequent cell:
//!
//! ```rust,no_run
//! # use charcoal::Chart;
//! # let df = polars::frame::DataFrame::empty();
//! let chart = Chart::scatter(&df).x("x").y("y").build()?;
//! # #[cfg(feature = "notebook")]
//! chart.display();
//! # Ok::<(), charcoal::CharcoalError>(())
//! ```
//!
//! # Repository
//!
//! <https://github.com/your-handle/charcoal>

mod dtype;
mod error;
mod normalize;
mod theme;
mod render;
mod charts;

#[cfg(feature = "notebook")]
mod display;

pub use charts::Chart;
pub use charts::{DashStyle, NullPolicy, Orientation};
pub use charts::area::FillMode;
pub use charts::box_plot::PointDisplay;
pub use charts::histogram::BinMethod;
pub use error::{CharcoalError, CharcoalWarning};
pub use theme::{ColorScale, Theme};