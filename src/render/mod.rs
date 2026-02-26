#![allow(dead_code)]

pub(crate) mod geometry;
pub(crate) mod axes;
pub(crate) mod html;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::theme::ThemeConfig;
use crate::render::geometry as geo;
use crate::render::axes::AxisOutput;

static CANVAS_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn unique_id() -> String {
    let n = CANVAS_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("cc_{:06x}", n)
}

#[derive(Debug, Clone)]
pub(crate) struct Margin {
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub left: u32,
}

impl Margin {

    pub(crate) fn new(top: u32, right: u32, bottom: u32, left: u32) -> Self {
        Self { top, right, bottom, left }
    }

    pub(crate) fn default_chart() -> Self {
        Self { top: 50, right: 30, bottom: 60, left: 70 }
    }
}

pub(crate) struct SvgCanvas {
    pub width: u32,
    pub height: u32,
    pub margin: Margin,
    pub theme: ThemeConfig,
    pub chart_id: String,
}

impl SvgCanvas {
    pub(crate) fn new(width: u32, height: u32, margin: Margin, theme: ThemeConfig) -> Self {
        Self {
            width,
            height,
            margin,
            theme,
            chart_id: unique_id(),
        }
    }

    pub(crate) fn plot_width(&self) -> f64 {
        (self.width as i64 - self.margin.left as i64 - self.margin.right as i64).max(0) as f64
    }

    pub(crate) fn plot_height(&self) -> f64 {
        (self.height as i64 - self.margin.top as i64 - self.margin.bottom as i64).max(0) as f64
    }

    pub(crate) fn plot_origin_x(&self) -> f64 {
        self.margin.left as f64
    }

    pub(crate) fn plot_origin_y(&self) -> f64 {
        self.margin.top as f64
    }

    pub(crate) fn render(
        &self,
        elements: Vec<String>,
        x_axis: AxisOutput,
        y_axis: AxisOutput,
        title: &str,
        x_label: &str,
        y_label: &str,
        legend: Option<Vec<(String, String)>>,
    ) -> String {
        let clip_id = format!("{}_clip", self.chart_id);
        let svg_id = &self.chart_id;

        let w = self.width;
        let h = self.height;
        let ox = self.plot_origin_x();
        let oy = self.plot_origin_y();
        let pw = self.plot_width();
        let ph = self.plot_height();

        let svg_open = format!(
            r#"<svg id="{svg_id}" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#,
        );
        let background = geo::rect(0.0, 0.0, w as f64, h as f64, self.theme.background, 0.0);
        let defs = format!(
            r#"<defs><clipPath id="{clip_id}"><rect x="{ox:.2}" y="{oy:.2}" width="{pw:.2}" height="{ph:.2}"/></clipPath></defs>"#,
        );
        let grids = format!("{}\n{}", x_axis.grid_lines, y_axis.grid_lines);
        let axes_svg = format!("{}\n{}", x_axis.axis_line, y_axis.axis_line);
        let data_group = geo::group(&elements, Some(&clip_id));
        let title_svg = if title.is_empty() {
            String::new()
        } else {
            geo::text(
                w as f64 / 2.0,
                self.margin.top as f64 * 0.6,
                title,
                "middle",
                self.theme.font_size_px + 4,
                self.theme.text_color,
                0.0,
            )
        };

        let x_label_svg = if x_label.is_empty() {
            String::new()
        } else {
            geo::text(
                ox + pw / 2.0,
                h as f64 - 6.0,
                x_label,
                "middle",
                self.theme.font_size_px,
                self.theme.text_color,
                0.0,
            )
        };

        let y_label_svg = if y_label.is_empty() {
            String::new()
        } else {
            let lx = self.margin.left as f64 * 0.35;
            let ly = oy + ph / 2.0;
            geo::text(lx, ly, y_label, "middle", self.theme.font_size_px, self.theme.text_color, -90.0)
        };

        let legend_svg = match legend {
            None => String::new(),
            Some(entries) => render_legend(&entries, ox + pw + 8.0, oy, &self.theme),
        };

        let parts = [
            svg_open.as_str(),
            &background,
            &defs,
            &grids,
            &axes_svg,
            &data_group,
            &title_svg,
            &x_label_svg,
            &y_label_svg,
            &legend_svg,
            "</svg>",
        ];

        parts.join("\n")
    }
}

fn render_legend(entries: &[(String, String)], x: f64, y_start: f64, theme: &ThemeConfig) -> String {
    let swatch_size = 12.0;
    let row_height = swatch_size + 4.0;
    let text_offset = swatch_size + 4.0;

    let mut parts = Vec::new();
    for (i, (label, color)) in entries.iter().enumerate() {
        let y = y_start + i as f64 * row_height;
        parts.push(geo::rect(x, y, swatch_size, swatch_size, color, 2.0));
        parts.push(geo::text(
            x + text_offset,
            y + swatch_size * 0.8,
            label,
            "start",
            theme.font_size_px,
            theme.text_color,
            0.0,
        ));
    }
    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{Theme, ThemeConfig};
    use crate::render::axes::AxisOutput;

    fn default_canvas() -> SvgCanvas {
        SvgCanvas::new(800, 600, Margin::default_chart(), ThemeConfig::from(&Theme::Default))
    }

    fn empty_axis() -> AxisOutput {
        AxisOutput {
            ticks: Vec::new(),
            axis_line: String::new(),
            grid_lines: String::new(),
        }
    }

    #[test]
    fn render_starts_with_svg_and_ends_with_close() {
        let canvas = default_canvas();
        let svg = canvas.render(vec![], empty_axis(), empty_axis(), "", "", "", None);
        assert!(svg.trim_start().starts_with("<svg"), "should start with <svg: {}", &svg[..50]);
        assert!(svg.trim_end().ends_with("</svg>"), "should end with </svg>");
    }

    #[test]
    fn render_contains_clip_path() {
        let canvas = default_canvas();
        let svg = canvas.render(vec![], empty_axis(), empty_axis(), "", "", "", None);
        assert!(svg.contains("<clipPath"), "missing <clipPath: found none");
    }

    #[test]
    fn render_includes_title_when_provided() {
        let canvas = default_canvas();
        let svg = canvas.render(vec![], empty_axis(), empty_axis(), "My Chart", "", "", None);
        assert!(svg.contains("My Chart"), "title not found in SVG");
    }

    #[test]
    fn render_omits_title_when_empty() {
        let canvas = default_canvas();
        let svg = canvas.render(vec![], empty_axis(), empty_axis(), "", "", "", None);
        assert!(!svg.contains("<text></text>"), "empty title text element present");
    }

    #[test]
    fn plot_width_subtracts_margins() {
        let canvas = SvgCanvas::new(
            800,
            600,
            Margin::new(50, 30, 60, 70),
            ThemeConfig::from(&Theme::Default),
        );
        assert!((canvas.plot_width() - (800.0 - 70.0 - 30.0)).abs() < 1e-9);
    }

    #[test]
    fn plot_height_subtracts_margins() {
        let canvas = SvgCanvas::new(
            800,
            600,
            Margin::new(50, 30, 60, 70),
            ThemeConfig::from(&Theme::Default),
        );
        assert!((canvas.plot_height() - (600.0 - 50.0 - 60.0)).abs() < 1e-9);
    }

    #[test]
    fn plot_origin_x_equals_left_margin() {
        let canvas = SvgCanvas::new(800, 600, Margin::new(50, 30, 60, 70), ThemeConfig::from(&Theme::Default));
        assert!((canvas.plot_origin_x() - 70.0).abs() < 1e-9);
    }

    #[test]
    fn plot_origin_y_equals_top_margin() {
        let canvas = SvgCanvas::new(800, 600, Margin::new(50, 30, 60, 70), ThemeConfig::from(&Theme::Default));
        assert!((canvas.plot_origin_y() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn unique_id_returns_different_values() {
        let id1 = unique_id();
        let id2 = unique_id();
        let id3 = unique_id();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn unique_id_starts_with_cc_prefix() {
        let id = unique_id();
        assert!(id.starts_with("cc_"), "id should start with cc_: {id}");
    }

    #[test]
    fn two_canvases_have_different_chart_ids() {
        let c1 = default_canvas();
        let c2 = default_canvas();
        assert_ne!(c1.chart_id, c2.chart_id);
    }

    #[test]
    fn render_legend_entries_appear_in_output() {
        let canvas = default_canvas();
        let legend = vec![
            ("Setosa".to_string(), "#E69F00".to_string()),
            ("Versicolor".to_string(), "#56B4E9".to_string()),
        ];
        let svg = canvas.render(vec![], empty_axis(), empty_axis(), "", "", "", Some(legend));
        assert!(svg.contains("Setosa"), "legend label Setosa missing");
        assert!(svg.contains("Versicolor"), "legend label Versicolor missing");
        assert!(svg.contains("#E69F00"), "legend color missing");
    }


    #[test]
    fn full_pipeline_produces_valid_svg() {
        use crate::render::axes::{
            LinearScale, AxisOrientation, nice_ticks, tick_labels_numeric,
            build_tick_marks, compute_axis,
        };
        use crate::render::geometry::circle;

        let canvas = default_canvas();
        let theme = ThemeConfig::from(&Theme::Default);

        let ox = canvas.plot_origin_x();
        let oy = canvas.plot_origin_y();
        let pw = canvas.plot_width();
        let ph = canvas.plot_height();

        let x_scale = LinearScale::new(0.0, 100.0, ox, ox + pw);
        let x_vals = nice_ticks(0.0, 100.0, 6);
        let x_labels = tick_labels_numeric(&x_vals);
        let x_ticks = build_tick_marks(&x_vals, &x_labels, &x_scale);
        let x_axis = compute_axis(&x_scale, &x_ticks, AxisOrientation::Horizontal,
            ox, oy, pw, ph, &theme);

        let y_scale = LinearScale::new(0.0, 50.0, oy + ph, oy);
        let y_vals = nice_ticks(0.0, 50.0, 5);
        let y_labels = tick_labels_numeric(&y_vals);
        let y_ticks = build_tick_marks(&y_vals, &y_labels, &y_scale);
        let y_axis = compute_axis(&y_scale, &y_ticks, AxisOrientation::Vertical,
            ox, oy, pw, ph, &theme);

        let elements: Vec<String> = [(25.0, 30.0), (50.0, 45.0), (75.0, 20.0)]
            .iter()
            .map(|&(xd, yd)| circle(x_scale.map(xd), y_scale.map(yd), 5.0, "#4e79a7", 0.8))
            .collect();

        let svg = canvas.render(
            elements, x_axis, y_axis,
            "Integration Test", "X Axis", "Y Axis",
            Some(vec![("Series A".to_string(), "#4e79a7".to_string())]),
        );

        assert!(svg.starts_with("<svg"),        "must start with <svg");
        assert!(svg.ends_with("</svg>"),        "must end with </svg>");
        assert!(svg.contains("<clipPath"),       "must contain <clipPath");
        assert!(svg.contains("Integration Test"), "title must appear");
        assert!(svg.contains("X Axis"),         "x-label must appear");
        assert!(svg.contains("Y Axis"),         "y-label must appear");
        assert!(svg.contains("Series A"),       "legend must appear");
        assert!(svg.contains("circle"),         "data points must appear");
    }

    #[test]
    fn html_wrapper_around_rendered_svg() {
        use crate::render::html::to_inline_html;

        let canvas = default_canvas();
        let svg = canvas.render(vec![], empty_axis(), empty_axis(), "HTML Test", "", "", None);
        let html = to_inline_html(&svg, "HTML Test");

        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("<title>HTML Test</title>"));
        assert!(html.contains("<svg ") || html.contains("<svg\n"));
        assert!(html.contains("</svg>"));
        let body_pos = html.rfind("</body>").expect("</body> not found");
        let html_pos = html.rfind("</html>").expect("</html> not found");
        assert!(body_pos < html_pos, "</body> must appear before </html>");
    }

    #[test]
    fn inverted_y_scale_maps_correctly() {
        use crate::render::axes::LinearScale;
        let scale = LinearScale::new(0.0, 100.0, 500.0, 50.0);
        assert!((scale.map(0.0)   - 500.0).abs() < 1e-9, "min should map to pixel bottom");
        assert!((scale.map(100.0) -  50.0).abs() < 1e-9, "max should map to pixel top");
        assert!((scale.map(50.0)  - 275.0).abs() < 1e-9, "midpoint should be mid-pixel");
    }

    #[test]
    fn write_html_for_browser_check() {
        use crate::render::axes::{
            LinearScale, AxisOrientation, nice_ticks, tick_labels_numeric,
            build_tick_marks, compute_axis,
        };
        use crate::render::geometry::circle;
        use crate::render::html::to_inline_html;

        let canvas = default_canvas();
        let theme = ThemeConfig::from(&Theme::Default);
        let ox = canvas.plot_origin_x();
        let oy = canvas.plot_origin_y();
        let pw = canvas.plot_width();
        let ph = canvas.plot_height();

        let x_scale = LinearScale::new(0.0, 10.0, ox, ox + pw);
        let x_vals  = nice_ticks(0.0, 10.0, 6);
        let x_ticks = build_tick_marks(&x_vals, &tick_labels_numeric(&x_vals), &x_scale);
        let x_axis  = compute_axis(&x_scale, &x_ticks, AxisOrientation::Horizontal, ox, oy, pw, ph, &theme);

        let y_scale = LinearScale::new(0.0, 10.0, oy + ph, oy);
        let y_vals  = nice_ticks(0.0, 10.0, 6);
        let y_ticks = build_tick_marks(&y_vals, &tick_labels_numeric(&y_vals), &y_scale);
        let y_axis  = compute_axis(&y_scale, &y_ticks, AxisOrientation::Vertical, ox, oy, pw, ph, &theme);

        let points = [(2.0,3.0),(4.0,7.0),(6.0,5.0),(8.0,9.0),(9.0,2.0)];
        let elements: Vec<String> = points.iter()
            .map(|&(xd,yd)| circle(x_scale.map(xd), y_scale.map(yd), 6.0, "#4e79a7", 0.8))
            .collect();

        let svg  = canvas.render(elements, x_axis, y_axis, "Phase 1 Render Test", "X", "Y",
            Some(vec![("Series A".to_string(), "#4e79a7".to_string())]));
        let html = to_inline_html(&svg, "charcoal render test");

        std::fs::create_dir_all("output").unwrap();
        std::fs::write("output/render_test.html", &html).unwrap();
        println!("\nWritten to output/render_test.html — open in your browser.");
    }
}