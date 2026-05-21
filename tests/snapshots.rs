use charcoal::{Chart, Theme, ColorScale, NullPolicy, FillMode, PointDisplay};
use polars::prelude::*;

/// Replace all `cc_XXXXXX` (6-hex-digit) IDs with `cc_000000` so snapshots
/// are stable regardless of which order tests run or how many canvases were
/// created before this one.
fn normalize_svg(svg: &str) -> String {
    let mut out  = String::with_capacity(svg.len());
    let mut rest = svg;
    while let Some(pos) = rest.find("cc_") {
        let tail = &rest[pos + 3..];
        if tail.len() >= 6 && tail[..6].chars().all(|c| c.is_ascii_hexdigit()) {
            out.push_str(&rest[..pos]);
            out.push_str("cc_000000");
            rest = &rest[pos + 9..];
        } else {
            out.push_str(&rest[..pos + 3]);
            rest = &rest[pos + 3..];
        }
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------------
// Fixtures — deterministic, hardcoded DataFrames, no file I/O or randomness
// ---------------------------------------------------------------------------

fn scatter_df() -> DataFrame {
    DataFrame::new(vec![
        Series::new("sepal_length", &[5.1f64, 4.9, 4.7, 6.4, 6.3, 5.8, 5.7]),
        Series::new("sepal_width",  &[3.5f64, 3.0, 3.2, 3.2, 3.3, 2.7, 2.8]),
        Series::new("species", &["setosa", "setosa", "setosa", "versicolor", "virginica", "versicolor", "virginica"]),
    ]).unwrap()
}

fn line_df() -> DataFrame {
    DataFrame::new(vec![
        Series::new("t", &[1.0f64, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        Series::new("v", &[10.0f64, 20.0, 15.0, 25.0, 18.0, 5.0, 12.0, 8.0, 17.0, 11.0]),
        Series::new("series", &["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]),
    ]).unwrap()
}

fn bar_df() -> DataFrame {
    DataFrame::new(vec![
        Series::new("category", &["Alpha", "Beta", "Gamma", "Delta"]),
        Series::new("count",    &[42.0f64, 28.0, 61.0, 35.0]),
        Series::new("group",    &["X", "X", "Y", "Y"]),
    ]).unwrap()
}

fn histogram_df() -> DataFrame {
    // 20 fixed float values approximating a bell distribution
    DataFrame::new(vec![
        Series::new("value", &[
            2.1f64, 2.4, 2.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.5,
            3.6, 3.6, 3.7, 3.8, 3.9, 4.0, 4.2, 4.5, 4.8, 5.0,
        ]),
    ]).unwrap()
}

fn heatmap_df() -> DataFrame {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut vs = Vec::new();
    for (r, row) in ["A", "B", "C", "D", "E"].iter().enumerate() {
        for (c, col) in ["V", "W", "X", "Y", "Z"].iter().enumerate() {
            xs.push(*col);
            ys.push(*row);
            vs.push((r * 5 + c) as f64 * 0.25);
        }
    }
    DataFrame::new(vec![
        Series::new("x_label", &xs),
        Series::new("y_label", &ys),
        Series::new("value",   &vs),
    ]).unwrap()
}

fn box_plot_df() -> DataFrame {
    DataFrame::new(vec![
        Series::new("group", &["Control", "Control", "Control", "Control", "Control",
                               "Treated", "Treated", "Treated", "Treated", "Treated"]),
        Series::new("value", &[2.1f64, 2.5, 2.9, 3.3, 3.7, 3.8, 4.2, 4.6, 5.0, 5.4]),
    ]).unwrap()
}

fn area_df() -> DataFrame {
    DataFrame::new(vec![
        Series::new("t", &[1.0f64, 2.0, 3.0, 4.0, 5.0]),
        Series::new("v", &[3.0f64, 5.0, 4.0, 7.0, 6.0]),
    ]).unwrap()
}

// ---------------------------------------------------------------------------
// 4.1.2 — Happy-path snapshot: one per chart type
// ---------------------------------------------------------------------------

#[test]
fn snapshot_scatter() {
    let df    = scatter_df();
    let chart = Chart::scatter(&df)
        .x("sepal_length")
        .y("sepal_width")
        .color_by("species")
        .title("Iris Scatter")
        .build()
        .unwrap();
    insta::assert_snapshot!("scatter", normalize_svg(chart.svg()));
}

#[test]
fn snapshot_line() {
    let df    = line_df();
    let chart = Chart::line(&df)
        .x("t")
        .y("v")
        .color_by("series")
        .title("Timeseries Line")
        .null_policy(NullPolicy::Skip)
        .build()
        .unwrap();
    insta::assert_snapshot!("line", normalize_svg(chart.svg()));
}

#[test]
fn snapshot_bar() {
    let df    = bar_df();
    let chart = Chart::bar(&df)
        .x("category")
        .y("count")
        .title("Bar Chart")
        .build()
        .unwrap();
    insta::assert_snapshot!("bar", normalize_svg(chart.svg()));
}

#[test]
fn snapshot_histogram() {
    let df    = histogram_df();
    let chart = Chart::histogram(&df)
        .x("value")
        .bins(10)
        .title("Histogram")
        .build()
        .unwrap();
    insta::assert_snapshot!("histogram", normalize_svg(chart.svg()));
}

#[test]
fn snapshot_heatmap() {
    let df    = heatmap_df();
    let chart = Chart::heatmap(&df)
        .x("x_label")
        .y("y_label")
        .z("value")
        .color_scale(ColorScale::Viridis)
        .annotate(true)
        .title("Heatmap Correlation")
        .build()
        .unwrap();
    insta::assert_snapshot!("heatmap", normalize_svg(chart.svg()));
}

#[test]
fn snapshot_box_plot() {
    let df    = box_plot_df();
    let chart = Chart::box_plot(&df)
        .x("group")
        .y("value")
        .points(PointDisplay::Outliers)
        .title("Box Plot Groups")
        .build()
        .unwrap();
    insta::assert_snapshot!("box_plot", normalize_svg(chart.svg()));
}

#[test]
fn snapshot_area() {
    let df    = area_df();
    let chart = Chart::area(&df)
        .x("t")
        .y("v")
        .fill_mode(FillMode::ToZero)
        .title("Area Chart")
        .build()
        .unwrap();
    insta::assert_snapshot!("area", normalize_svg(chart.svg()));
}

// ---------------------------------------------------------------------------
// 4.1.6 — Edge-case snapshots
// ---------------------------------------------------------------------------

/// A chart with no title must not include a <text> title element.
#[test]
fn snapshot_scatter_no_title() {
    let df    = scatter_df();
    let chart = Chart::scatter(&df)
        .x("sepal_length")
        .y("sepal_width")
        .build()
        .unwrap();
    let svg = chart.svg();
    insta::assert_snapshot!("scatter_no_title", normalize_svg(svg));
}

/// Dark theme must use a dark background colour, distinct from the Default theme.
#[test]
fn snapshot_scatter_dark_theme() {
    let df    = scatter_df();
    let chart = Chart::scatter(&df)
        .x("sepal_length")
        .y("sepal_width")
        .theme(Theme::Dark)
        .title("Dark Theme")
        .build()
        .unwrap();
    let svg = chart.svg();
    // Confirm the dark background is present.
    assert!(svg.contains("#1E1E1E"), "Dark theme must include background #1E1E1E");
    insta::assert_snapshot!("scatter_dark_theme", normalize_svg(svg));
}

/// Palette wrapping: 9 categories with an 8-colour palette.
/// All 9 labels must appear and the build must not panic.
#[test]
fn snapshot_scatter_palette_wrap() {
    let cats: Vec<&str> = vec!["a","b","c","d","e","f","g","h","i"];
    let xs: Vec<f64>   = (0..9).map(|i| i as f64).collect();
    let ys: Vec<f64>   = xs.iter().map(|x| x * 1.5).collect();
    let df = DataFrame::new(vec![
        Series::new("x",   &xs),
        Series::new("y",   &ys),
        Series::new("cat", &cats),
    ]).unwrap();
    let chart = Chart::scatter(&df)
        .x("x")
        .y("y")
        .color_by("cat")
        .title("Palette Wrap")
        .build()
        .unwrap();
    let svg = chart.svg();
    for label in &cats {
        assert!(svg.contains(label), "legend must contain category '{label}'");
    }
    insta::assert_snapshot!("scatter_palette_wrap", normalize_svg(svg));
}
