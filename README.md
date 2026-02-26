# charcoal

A declarative, DataFrame-native chart library for Polars. No browser. No Python. No C FFI.
```toml
[dependencies]
charcoal = "0.1"
```

## Quickstart
```rust
use charcoal::{Chart, Theme};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = CsvReader::from_path("iris.csv")?.finish()?;

    let chart = Chart::scatter(&df)
        .x("sepal_length")
        .y("sepal_width")
        .color_by("species")
        .title("Iris Dataset")
        .theme(Theme::Default)
        .build()?;

    chart.save_svg("iris.svg")?;
    chart.save_html("iris.html")?;
    chart.save_png("iris.png")?; // requires `static` feature

    Ok(())
}
```

## Chart Types

| Chart | Method | Required Columns |
|-------|--------|-----------------|
| Scatter | `Chart::scatter(&df)` | `.x()`, `.y()` |
| Line | `Chart::line(&df)` | `.x()`, `.y()` |
| Bar | `Chart::bar(&df)` | `.x()`, `.y()` |
| Histogram | `Chart::histogram(&df)` | `.x()` |
| Heatmap | `Chart::heatmap(&df)` | `.x()`, `.y()`, `.z()` |
| Box Plot | `Chart::box_plot(&df)` | `.x()`, `.y()` |
| Area | `Chart::area(&df)` | `.x()`, `.y()` |

## Output Formats

| Format | Method | Feature Flag |
|--------|--------|-------------|
| SVG string | `chart.svg()` | none |
| SVG file | `chart.save_svg("out.svg")` | none |
| Standalone HTML | `chart.save_html("out.html")` | none |
| PNG / JPEG / WEBP | `chart.save_png("out.png")` | `static` |
| evcxr notebook | `chart.display()` | `notebook` |

## Feature Flags

| Feature | Enables | Extra Dependencies |
|---------|---------|--------------------|
| `static` | PNG/JPEG/WEBP raster export | `resvg` (pure Rust) |
| `notebook` | Inline display in evcxr | `evcxr_runtime` |
| `ndarray` | `Array2<f64>` input for heatmaps | `ndarray` |
| `interactive` | Plotly.js interactive HTML export | none |

## Themes
```rust
Theme::Default     // clean light theme
Theme::Dark        // dark background
Theme::Minimal     // no gridlines, minimal chrome
Theme::Colorblind  // Wong 8-color palette
```

## Error Quality

charcoal errors tell you what went wrong, where, and what to do next:
```
CharcoalError::ColumnNotFound
  column "sepal_lenght" not found
  Did you mean: sepal_length
  Available: sepal_length, sepal_width, petal_length, petal_width, species
```

## Null Handling

Every column role has a documented null policy. Nulls are never silently
dropped without a warning. Access warnings after rendering:
```rust
let chart = Chart::scatter(&df)
    .x("sepal_length")
    .y("sepal_width")
    .build()?;

for warning in chart.warnings() {
    eprintln!("warning: {warning}");
}
```

## Row Limits

- Above 500k rows: warning emitted, scatter points subsampled
- Above 1M rows: `.build()` returns `Err(CharcoalError::DataTooLarge)`

Configure the limit via the builder:
```rust
Chart::scatter(&df)
    .x("x")
    .y("y")
    .row_limit(2_000_000)
    .build()?;
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option.