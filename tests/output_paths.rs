/// Integration tests for all output paths across all seven chart types (Phase 3.4).
///
/// These tests exercise `save_svg`, `save_html`, `save_png` (feature-gated), and
/// `chart.svg()` end-to-end using small in-process DataFrames.
use charcoal::{Chart, CharcoalWarning};
use polars::prelude::*;

// ---------------------------------------------------------------------------
// Minimal DataFrame helpers
// ---------------------------------------------------------------------------

fn numeric_df() -> DataFrame {
    df!(
        "x" => &[1.0f64, 2.0, 3.0, 4.0, 5.0],
        "y" => &[2.0f64, 4.0, 1.0, 3.0, 5.0]
    )
    .unwrap()
}

fn categorical_df() -> DataFrame {
    df!(
        "category" => &["A", "B", "C", "D"],
        "value"    => &[10.0f64, 20.0, 15.0, 25.0]
    )
    .unwrap()
}

fn histogram_df() -> DataFrame {
    df!("x" => &[1.0f64, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        .unwrap()
}

fn heatmap_df() -> DataFrame {
    df!(
        "row" => &["A", "A", "B", "B"],
        "col" => &["X", "Y", "X", "Y"],
        "val" => &[1.0f64, 2.0, 3.0, 4.0]
    )
    .unwrap()
}

fn box_plot_df() -> DataFrame {
    df!(
        "group" => &["A", "A", "A", "B", "B", "B"],
        "value" => &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Helpers that build one chart of each type
// ---------------------------------------------------------------------------

fn build_scatter() -> Chart {
    Chart::scatter(&numeric_df()).x("x").y("y").build().unwrap()
}

fn build_line() -> Chart {
    Chart::line(&numeric_df()).x("x").y("y").build().unwrap()
}

fn build_bar() -> Chart {
    Chart::bar(&categorical_df()).x("category").y("value").build().unwrap()
}

fn build_histogram() -> Chart {
    Chart::histogram(&histogram_df()).x("x").build().unwrap()
}

fn build_heatmap() -> Chart {
    Chart::heatmap(&heatmap_df())
        .x("row")
        .y("col")
        .z("val")
        .build()
        .unwrap()
}

fn build_box_plot() -> Chart {
    Chart::box_plot(&box_plot_df())
        .x("group")
        .y("value")
        .build()
        .unwrap()
}

fn build_area() -> Chart {
    Chart::area(&numeric_df()).x("x").y("y").build().unwrap()
}

// ---------------------------------------------------------------------------
// 3.4.2 — save_svg for all seven chart types
// ---------------------------------------------------------------------------

#[test]
fn scatter_save_svg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scatter.svg");
    build_scatter().save_svg(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.trim_start().starts_with("<svg"), "expected SVG start");
    assert!(contents.trim_end().ends_with("</svg>"), "expected SVG end");
}

#[test]
fn line_save_svg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("line.svg");
    build_line().save_svg(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.trim_start().starts_with("<svg"));
    assert!(contents.trim_end().ends_with("</svg>"));
}

#[test]
fn bar_save_svg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bar.svg");
    build_bar().save_svg(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.trim_start().starts_with("<svg"));
    assert!(contents.trim_end().ends_with("</svg>"));
}

#[test]
fn histogram_save_svg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("histogram.svg");
    build_histogram().save_svg(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.trim_start().starts_with("<svg"));
    assert!(contents.trim_end().ends_with("</svg>"));
}

#[test]
fn heatmap_save_svg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heatmap.svg");
    build_heatmap().save_svg(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.trim_start().starts_with("<svg"));
    assert!(contents.trim_end().ends_with("</svg>"));
}

#[test]
fn box_plot_save_svg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("box_plot.svg");
    build_box_plot().save_svg(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.trim_start().starts_with("<svg"));
    assert!(contents.trim_end().ends_with("</svg>"));
}

#[test]
fn area_save_svg() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("area.svg");
    build_area().save_svg(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.trim_start().starts_with("<svg"));
    assert!(contents.trim_end().ends_with("</svg>"));
}

// ---------------------------------------------------------------------------
// 3.4.3 — save_html for all seven chart types
// ---------------------------------------------------------------------------

#[test]
fn scatter_save_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scatter.html");
    let chart = build_scatter();
    let svg = chart.svg().to_string();
    chart.save_html(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<!DOCTYPE html>"));
    assert!(contents.contains(&svg));
}

#[test]
fn line_save_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("line.html");
    let chart = build_line();
    let svg = chart.svg().to_string();
    chart.save_html(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<!DOCTYPE html>"));
    assert!(contents.contains(&svg));
}

#[test]
fn bar_save_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bar.html");
    let chart = build_bar();
    let svg = chart.svg().to_string();
    chart.save_html(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<!DOCTYPE html>"));
    assert!(contents.contains(&svg));
}

#[test]
fn histogram_save_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("histogram.html");
    let chart = build_histogram();
    let svg = chart.svg().to_string();
    chart.save_html(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<!DOCTYPE html>"));
    assert!(contents.contains(&svg));
}

#[test]
fn heatmap_save_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heatmap.html");
    let chart = build_heatmap();
    let svg = chart.svg().to_string();
    chart.save_html(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<!DOCTYPE html>"));
    assert!(contents.contains(&svg));
}

#[test]
fn box_plot_save_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("box_plot.html");
    let chart = build_box_plot();
    let svg = chart.svg().to_string();
    chart.save_html(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<!DOCTYPE html>"));
    assert!(contents.contains(&svg));
}

#[test]
fn area_save_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("area.html");
    let chart = build_area();
    let svg = chart.svg().to_string();
    chart.save_html(path.to_str().unwrap()).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("<!DOCTYPE html>"));
    assert!(contents.contains(&svg));
}

// ---------------------------------------------------------------------------
// 3.4.4 — save_png for all seven chart types (feature = "static")
// ---------------------------------------------------------------------------

#[cfg(feature = "static")]
const PNG_MAGIC: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

#[test]
#[cfg(feature = "static")]
fn scatter_save_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scatter.png");
    build_scatter().save_png(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], &PNG_MAGIC);
}

#[test]
#[cfg(feature = "static")]
fn line_save_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("line.png");
    build_line().save_png(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], &PNG_MAGIC);
}

#[test]
#[cfg(feature = "static")]
fn bar_save_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bar.png");
    build_bar().save_png(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], &PNG_MAGIC);
}

#[test]
#[cfg(feature = "static")]
fn histogram_save_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("histogram.png");
    build_histogram().save_png(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], &PNG_MAGIC);
}

#[test]
#[cfg(feature = "static")]
fn heatmap_save_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heatmap.png");
    build_heatmap().save_png(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], &PNG_MAGIC);
}

#[test]
#[cfg(feature = "static")]
fn box_plot_save_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("box_plot.png");
    build_box_plot().save_png(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], &PNG_MAGIC);
}

#[test]
#[cfg(feature = "static")]
fn area_save_png() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("area.png");
    build_area().save_png(path.to_str().unwrap()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], &PNG_MAGIC);
}

// ---------------------------------------------------------------------------
// 3.4.5 — chart.svg() returns non-empty string for all seven chart types
// ---------------------------------------------------------------------------

#[test]
fn all_chart_types_produce_nonempty_svg() {
    assert!(!build_scatter().svg().is_empty(), "scatter svg is empty");
    assert!(!build_line().svg().is_empty(),    "line svg is empty");
    assert!(!build_bar().svg().is_empty(),     "bar svg is empty");
    assert!(!build_histogram().svg().is_empty(), "histogram svg is empty");
    assert!(!build_heatmap().svg().is_empty(), "heatmap svg is empty");
    assert!(!build_box_plot().svg().is_empty(), "box_plot svg is empty");
    assert!(!build_area().svg().is_empty(),    "area svg is empty");
}

// ---------------------------------------------------------------------------
// 3.4.6 — chart.warnings() accumulation
// ---------------------------------------------------------------------------

#[test]
fn scatter_nulls_in_y_produce_warning() {
    let df = df!(
        "x" => &[1.0f64, 2.0, 3.0, 4.0, 5.0],
        "y" => &[Some(1.0f64), None, Some(3.0), Some(4.0), Some(5.0)]
    )
    .unwrap();
    let chart = Chart::scatter(&df).x("x").y("y").build().unwrap();
    assert!(
        !chart.warnings().is_empty(),
        "expected at least one warning for null y values"
    );
    assert!(
        chart
            .warnings()
            .iter()
            .any(|w| matches!(w, CharcoalWarning::NullsSkipped { col, .. } if col == "y")),
        "expected NullsSkipped warning for column 'y'"
    );
}
