pub trait Display {
    fn evcxr_display(&self);
}
pub fn mime_type<S: Into<String>>(mime_type: S) -> ContentMimeType;
impl ContentMimeType {
    pub fn text<S: AsRef<str>>(self, text: S);
}

//! evcxr inline display for [`Chart`].
//!
//! Compiled only when the `notebook` feature is active. Implements the
//! `evcxr_runtime::Display` trait so `chart.display()` renders the SVG inline
//! in a Jupyter / evcxr notebook output cell.

#![cfg(feature = "notebook")]

use crate::charts::Chart;

/// Renders [`Chart`] inline in an evcxr notebook cell as `image/svg+xml`.
///
/// The SVG string already lives on `self`; `mime_type(..).text(..)` borrows it
/// via `AsRef<str>` and emits the evcxr protocol markers without copying.
impl evcxr_runtime::Display for Chart {
    fn evcxr_display(&self) {
        evcxr_runtime::mime_type("image/svg+xml").text(&self.svg);
    }
}