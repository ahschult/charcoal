pub mod area;
pub mod bar;
pub mod box_plot;
pub mod heatmap;
pub mod histogram;
pub mod line;
pub mod scatter;

use std::fs;
use polars::frame::DataFrame;
use crate::error::{CharcoalError, CharcoalWarning};

impl Chart {
    /// Begin building a scatter chart from `df`.
    pub fn scatter(df: &DataFrame) -> scatter::ScatterBuilder<'_> {
        scatter::ScatterBuilder::new(df)
    }

    /// Begin building a line chart from `df`.
    pub fn line(df: &DataFrame) -> line::LineBuilder<'_> {
        line::LineBuilder::new(df)
    }

    /// Begin building a bar chart from `df`.
    pub fn bar(df: &DataFrame) -> bar::BarBuilder<'_> {
        bar::BarBuilder::new(df)
    }

    /// Begin building a histogram from `df`.
    pub fn histogram(df: &DataFrame) -> histogram::HistogramBuilder<'_> {
        histogram::HistogramBuilder::new(df)
    }

    /// Begin building a heatmap from `df`.
    pub fn heatmap(df: &DataFrame) -> heatmap::HeatmapBuilder<'_> {
        heatmap::HeatmapBuilder::new(df)
    }

    /// Begin building a box plot from `df`.
    pub fn box_plot(df: &DataFrame) -> box_plot::BoxPlotBuilder<'_> {
        box_plot::BoxPlotBuilder::new(df)
    }

    /// Begin building an area chart from `df`.
    pub fn area(df: &DataFrame) -> area::AreaBuilder<'_> {
        area::AreaBuilder::new(df)
    }
}

/// How null y-values are handled in connected series (Line, Area).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NullPolicy {
    /// Break the line at nulls, leaving a gap. Default.
    #[default]
    Skip,
    /// Fill gaps by linear interpolation between surrounding non-null values.
    Interpolate,
}

/// Dash style for line and area series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashStyle {
    #[default]
    Solid,
    /// `stroke-dasharray="6 3"`
    Dashed,
    /// `stroke-dasharray="2 2"`
    Dotted,
}

impl DashStyle {
    /// SVG `stroke-dasharray` value, or `None` for solid lines.
    pub(crate) fn stroke_dasharray(self) -> Option<&'static str> {
        match self {
            DashStyle::Solid => None,
            DashStyle::Dashed => Some("6 3"),
            DashStyle::Dotted => Some("2 2"),
        }
    }
}

/// The fully-rendered output of a `.build()` call.
///
/// ```rust,no_run
/// # use charcoal::Chart;
/// # let df = polars::frame::DataFrame::empty();
/// let chart = Chart::scatter(&df)
///     .x("sepal_length")
///     .y("sepal_width")
///     .build()?;
/// chart.save_svg("iris.svg")?;
/// # Ok::<(), charcoal::CharcoalError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Chart {
    pub(crate) svg: String,
    pub(crate) warnings: Vec<CharcoalWarning>,
    pub(crate) title: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Chart {
    /// The rendered SVG string.
    pub fn svg(&self) -> &str {
        &self.svg
    }

    /// Warnings accumulated during rendering. Empty slice means a clean build.
    pub fn warnings(&self) -> &[CharcoalWarning] {
        &self.warnings
    }

    /// Chart width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Chart height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Write the SVG to `path` (UTF-8, overwrites if exists).
    ///
    /// # Errors
    /// [`CharcoalError::Io`] if the path is not writable or its parent directory does not exist.
    pub fn save_svg(&self, path: &str) -> Result<(), CharcoalError> {
        fs::write(path, &self.svg)?;
        Ok(())
    }

    /// Write an HTML document embedding the SVG inline to `path`.
    ///
    /// The file has no external dependencies and can be opened directly in any browser.
    ///
    /// # Errors
    /// [`CharcoalError::Io`] if the path is not writable or its parent directory does not exist.
    pub fn save_html(&self, path: &str) -> Result<(), CharcoalError> {
        fs::write(path, to_inline_html(&self.svg, &self.title))?;
        Ok(())
    }

    /// Write a PNG raster image to `path`.
    ///
    /// Implemented in Phase 3 via `resvg` + `tiny-skia` (requires the `static` feature).
    /// The method exists now so call sites compile; it returns an error until Phase 3 is complete.
    ///
    /// # Errors
    /// [`CharcoalError::RenderError`] until the Phase 3 raster backend is in place.
    pub fn save_png(&self, _path: &str) -> Result<(), CharcoalError> {
        // Phase 3: drive resvg here when the `static` feature is active.
        #[cfg(feature = "static")]
        {}
        Err(CharcoalError::RenderError(
            "PNG export requires the `static` feature and is implemented in Phase 3. \
             Add `features = [\"static\"]` to your Cargo dependency once Phase 3 is released."
                .to_string(),
        ))
    }

    /// Display the chart inline in an evcxr / Jupyter notebook cell.
    ///
    /// Requires the `notebook` feature. No-op stub until Phase 3.
    #[cfg(feature = "notebook")]
    pub fn display(&self) {
        // Phase 3: evcxr_display::SVGWrapper(self.svg.clone()).display();
        let _ = self;
    }
}

/// Minimal self-contained HTML document with the SVG embedded inline.
///
/// `&`, `<`, and `>` in `title` are escaped. The SVG body is embedded verbatim.
pub(crate) fn to_inline_html(svg: &str, title: &str) -> String {
    let safe_title = title
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{safe_title}</title>
  <style>
    body {{
      margin: 0;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      background: #ffffff;
      padding: 1rem;
      box-sizing: border-box;
    }}
    svg {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
{svg}
</body>
</html>
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chart(svg: &str, title: &str) -> Chart {
        Chart {
            svg: svg.to_string(),
            warnings: vec![],
            title: title.to_string(),
            width: 800,
            height: 500,
        }
    }

    fn make_chart_with_warnings(svg: &str, title: &str, warnings: Vec<CharcoalWarning>) -> Chart {
        Chart { svg: svg.to_string(), warnings, title: title.to_string(), width: 800, height: 500 }
    }

    #[test]
    fn svg_returns_stored_string() {
        assert_eq!(make_chart("<svg/>", "T").svg(), "<svg/>");
    }

    #[test]
    fn svg_is_unmodified() {
        let raw = r#"<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500"></svg>"#;
        assert_eq!(make_chart(raw, "T").svg(), raw);
    }

    #[test]
    fn warnings_empty_by_default() {
        assert!(make_chart("<svg/>", "T").warnings().is_empty());
    }

    #[test]
    fn warnings_nulls_skipped() {
        let chart = make_chart_with_warnings(
            "<svg/>", "T",
            vec![CharcoalWarning::NullsSkipped { col: "y".to_string(), count: 3 }],
        );
        assert_eq!(chart.warnings().len(), 1);
        match &chart.warnings()[0] {
            CharcoalWarning::NullsSkipped { col, count } => {
                assert_eq!(col, "y");
                assert_eq!(*count, 3);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn warnings_rows_subsampled() {
        let chart = make_chart_with_warnings(
            "<svg/>", "T",
            vec![CharcoalWarning::RowsSubsampled { original: 1_000_000, rendered: 500_000 }],
        );
        match &chart.warnings()[0] {
            CharcoalWarning::RowsSubsampled { original, rendered } => {
                assert_eq!(*original, 1_000_000);
                assert_eq!(*rendered, 500_000);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn warnings_multiple() {
        let chart = make_chart_with_warnings(
            "<svg/>", "T",
            vec![
                CharcoalWarning::NullsSkipped { col: "x".to_string(), count: 1 },
                CharcoalWarning::NullsSkipped { col: "y".to_string(), count: 2 },
                CharcoalWarning::RowsSubsampled { original: 600_000, rendered: 500_000 },
            ],
        );
        assert_eq!(chart.warnings().len(), 3);
    }

    #[test]
    fn dimensions() {
        let chart = make_chart("<svg/>", "T");
        assert_eq!(chart.width(), 800);
        assert_eq!(chart.height(), 500);
    }

    #[test]
    fn dimensions_custom() {
        let chart = Chart {
            svg: "<svg/>".to_string(), warnings: vec![],
            title: "T".to_string(), width: 1200, height: 900,
        };
        assert_eq!(chart.width(), 1200);
        assert_eq!(chart.height(), 900);
    }

    #[test]
    fn save_svg_writes_exact_contents() {
        let svg = "<svg><text>hello</text></svg>";
        let path = "/tmp/charcoal_test_save_svg.svg";
        make_chart(svg, "T").save_svg(path).unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), svg);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn save_svg_overwrites() {
        let path = "/tmp/charcoal_test_save_svg_overwrite.svg";
        std::fs::write(path, "old content").unwrap();
        make_chart("<svg/>", "T").save_svg(path).unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "<svg/>");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn save_svg_bad_path_returns_io_error() {
        let result = make_chart("<svg/>", "T").save_svg("/no/such/dir/chart.svg");
        assert!(matches!(result.unwrap_err(), CharcoalError::Io(_)));
    }

    #[test]
    fn save_html_writes_valid_document() {
        let path = "/tmp/charcoal_test_save_html.html";
        make_chart("<svg/>", "My Chart").save_html(path).unwrap();
        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("<!DOCTYPE html>"));
        assert!(contents.contains("<title>My Chart</title>"));
        assert!(contents.contains("<svg/>"));
        assert!(contents.contains(r#"charset="UTF-8""#));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn save_html_bad_path_returns_io_error() {
        let result = make_chart("<svg/>", "T").save_html("/no/such/dir/chart.html");
        assert!(matches!(result.unwrap_err(), CharcoalError::Io(_)));
    }

    #[test]
    fn save_png_stub_returns_render_error() {
        let result = make_chart("<svg/>", "T").save_png("/tmp/should_not_exist.png");
        match result.unwrap_err() {
            CharcoalError::RenderError(msg) => {
                assert!(msg.contains("static"));
                assert!(msg.to_lowercase().contains("phase 3"));
            }
            other => panic!("expected RenderError, got {other:?}"),
        }
        assert!(!std::path::Path::new("/tmp/should_not_exist.png").exists());
    }

    #[test]
    fn html_escapes_title_special_chars() {
        let html = to_inline_html("<svg/>", "A & B <Chart>");
        assert!(html.contains("A &amp; B &lt;Chart&gt;"));
        assert!(!html.contains("A & B <Chart>"));
    }

    #[test]
    fn html_escapes_ampersand_not_in_title_element() {
        let html = to_inline_html("<svg/>", "Cats & Dogs");
        let title_content = &html[html.find("<title>").unwrap()..html.find("</title>").unwrap()];
        assert!(!title_content.contains("Cats & Dogs"));
    }

    #[test]
    fn html_embeds_svg_verbatim() {
        let svg = r#"<svg width="800" height="500"><rect/></svg>"#;
        assert!(to_inline_html(svg, "T").contains(svg));
    }

    #[test]
    fn html_structure() {
        let html = to_inline_html("<svg/>", "T");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<html"));
        assert!(html.contains(r#"charset="UTF-8""#));
        assert!(html.contains("<body>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn html_plain_title_unchanged() {
        assert!(to_inline_html("<svg/>", "Simple Title").contains("<title>Simple Title</title>"));
    }

    #[test]
    fn dash_style_defaults_and_dasharray() {
        assert_eq!(DashStyle::default(), DashStyle::Solid);
        assert_eq!(DashStyle::Solid.stroke_dasharray(), None);
        assert_eq!(DashStyle::Dashed.stroke_dasharray(), Some("6 3"));
        assert_eq!(DashStyle::Dotted.stroke_dasharray(), Some("2 2"));
    }

    #[test]
    fn null_policy_default_is_skip() {
        assert_eq!(NullPolicy::default(), NullPolicy::Skip);
    }
}
