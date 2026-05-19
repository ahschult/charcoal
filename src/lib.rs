//! # Notebook Usage
//!
//! To display charts inline in an [evcxr](https://github.com/evcxr/evcxr) Jupyter
//! notebook, enable the `notebook` feature:
//!
//! ```toml
//! # Cargo.toml
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

mod dtype;
mod error;
mod normalize;
mod theme;
mod render;
mod charts;

#[cfg(feature = "notebook")]
mod display;

pub use charts::Chart;
pub use charts::{DashStyle, NullPolicy};
pub use error::{CharcoalError, CharcoalWarning};
pub use theme::{ColorScale, Theme};

#[cfg(feature = "notebook")]
mod display;