#![allow(dead_code)]

use std::collections::HashSet;

use crate::charts::{Chart, DashStyle};
use crate::error::{CharcoalError, CharcoalWarning};
use crate::render::{
    Margin, SvgCanvas,
    axes::{
        AxisOrientation, LinearScale, build_tick_marks, compute_axis,
        nice_ticks, tick_labels_numeric,
    },
    geometry,
};
use crate::theme::{Theme, ThemeConfig};

// ---------------------------------------------------------------------------
// Internal enums
// ---------------------------------------------------------------------------

/// Identifies the chart type a [`Layer`] came from. Used for compatibility checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    Scatter,
    Line,
    Bar,
    Area,
    HLine,
    VLine,
    Annotation,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisType {
    Numeric,
    Temporal,
    Categorical,
}

pub(crate) struct LayerSpec {
    pub kind:                 LayerKind,
    pub x_axis_type:          AxisType,
    pub y_axis_type:          AxisType,
    pub x_min:                f64,
    pub x_max:                f64,
    pub y_min:                f64,
    pub y_max:                f64,
    /// `true` for data layers; `false` for reference layers (hline/vline/annotation).
    pub contributes_to_range: bool,
    /// Render closure: given resolved x/y scales, returns `(svg_elements, warnings)`.
    pub render: Box<dyn FnOnce(&LinearScale, &LinearScale) -> (Vec<String>, Vec<CharcoalWarning>)>,
    /// Legend entries `(label, hex_color)` contributed by this layer.
    pub legend_entries: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Layer — public wrapper
// ---------------------------------------------------------------------------

/// A single composable layer for use in [`LayeredChart`].
///
/// Wraps a deferred `prepare` closure that validates the layer's data and produces
/// a [`LayerSpec`] when [`LayeredChart::build()`] is called.
pub struct Layer {
    pub(crate) prepare: Box<dyn FnOnce() -> Result<LayerSpec, CharcoalError>>,
}

impl Layer {
    // -----------------------------------------------------------------------
    // Data layer entry points
    // -----------------------------------------------------------------------

    /// Start a scatter layer (same builder chain as `Chart::scatter`).
    pub fn scatter(df: &polars::frame::DataFrame) -> crate::chart::scatter::ScatterBuilder<'_> {
        crate::chart::scatter::ScatterBuilder::new(df)
    }

    /// Start a line layer (same builder chain as `Chart::line`).
    pub fn line(df: &polars::frame::DataFrame) -> crate::chart::line::LineBuilder<'_> {
        crate::chart::line::LineBuilder::new(df)
    }

    /// Start a bar layer (same builder chain as `Chart::bar`).
    pub fn bar(df: &polars::frame::DataFrame) -> crate::chart::bar::BarBuilder<'_> {
        crate::chart::bar::BarBuilder::new(df)
    }

    /// Start an area layer (same builder chain as `Chart::area`).
    pub fn area(df: &polars::frame::DataFrame) -> crate::chart::area::AreaBuilder<'_> {
        crate::chart::area::AreaBuilder::new(df)
    }

    // -----------------------------------------------------------------------
    // Reference layer entry points
    // -----------------------------------------------------------------------

    /// Draw a horizontal line across the full plot width at y data-space value `y`.
    pub fn hline(y: f64) -> HLineBuilder {
        HLineBuilder { y, color: None, width: 1.5, dash: DashStyle::Dashed, label: None }
    }

    /// Draw a vertical line across the full plot height at x data-space value `x`.
    pub fn vline(x: f64) -> VLineBuilder {
        VLineBuilder { x, color: None, width: 1.5, dash: DashStyle::Dashed, label: None }
    }

    /// Place a text label at data-space coordinate `(x, y)`.
    pub fn annotation(x: f64, y: f64, text: &str) -> AnnotationBuilder {
        AnnotationBuilder {
            x, y,
            text:     text.to_string(),
            color:    None,
            size:     12,
            anchor:   TextAnchor::Start,
            offset_x: 4.0,
            offset_y: -4.0,
        }
    }
}

// ---------------------------------------------------------------------------
// HLineBuilder
// ---------------------------------------------------------------------------

/// Builder for a horizontal reference line. Created by [`Layer::hline`].
pub struct HLineBuilder {
    y:     f64,
    color: Option<String>,
    width: f64,
    dash:  DashStyle,
    label: Option<String>,
}

impl HLineBuilder {
    /// Line color (hex). Default: `"#888888"`.
    pub fn color(mut self, hex: &str) -> Self { self.color = Some(hex.to_string()); self }
    /// Stroke width in pixels. Default: `1.5`.
    pub fn width(mut self, px: f64) -> Self { self.width = px; self }
    /// Dash style. Default: [`DashStyle::Dashed`].
    pub fn dash(mut self, style: DashStyle) -> Self { self.dash = style; self }
    /// Add a legend entry with this label.
    pub fn label(mut self, text: &str) -> Self { self.label = Some(text.to_string()); self }

    /// Consume this builder and produce a [`Layer`].
    pub fn into_layer(self) -> Layer {
        Layer {
            prepare: Box::new(move || {
                let color_str = self.color.clone().unwrap_or_else(|| "#888888".to_string());
                let legend_entries = self.label.as_ref()
                    .map(|l| vec![(l.clone(), color_str.clone())])
                    .unwrap_or_default();
                let (y, width, dash, color_owned) = (self.y, self.width, self.dash, color_str);
                Ok(LayerSpec {
                    kind:                 LayerKind::HLine,
                    x_axis_type:          AxisType::Numeric,
                    y_axis_type:          AxisType::Numeric,
                    x_min:                0.0,
                    x_max:                0.0,
                    y_min:                y,
                    y_max:                y,
                    contributes_to_range: false,
                    legend_entries,
                    render: Box::new(move |x_scale, y_scale| {
                        let py   = y_scale.map(y);
                        let dash = dash.stroke_dasharray().unwrap_or("");
                        let elem = geometry::dashed_line(
                            x_scale.pixel_min, py, x_scale.pixel_max, py,
                            &color_owned, width, dash,
                        );
                        (vec![elem], vec![])
                    }),
                })
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// VLineBuilder
// ---------------------------------------------------------------------------

/// Builder for a vertical reference line. Created by [`Layer::vline`].
pub struct VLineBuilder {
    x:     f64,
    color: Option<String>,
    width: f64,
    dash:  DashStyle,
    label: Option<String>,
}

impl VLineBuilder {
    /// Line color (hex). Default: `"#888888"`.
    pub fn color(mut self, hex: &str) -> Self { self.color = Some(hex.to_string()); self }
    /// Stroke width in pixels. Default: `1.5`.
    pub fn width(mut self, px: f64) -> Self { self.width = px; self }
    /// Dash style. Default: [`DashStyle::Dashed`].
    pub fn dash(mut self, style: DashStyle) -> Self { self.dash = style; self }
    /// Add a legend entry with this label.
    pub fn label(mut self, text: &str) -> Self { self.label = Some(text.to_string()); self }

    /// Consume this builder and produce a [`Layer`].
    pub fn into_layer(self) -> Layer {
        Layer {
            prepare: Box::new(move || {
                let color_str = self.color.clone().unwrap_or_else(|| "#888888".to_string());
                let legend_entries = self.label.as_ref()
                    .map(|l| vec![(l.clone(), color_str.clone())])
                    .unwrap_or_default();
                let (x, width, dash, color_owned) = (self.x, self.width, self.dash, color_str);
                Ok(LayerSpec {
                    kind:                 LayerKind::VLine,
                    x_axis_type:          AxisType::Numeric,
                    y_axis_type:          AxisType::Numeric,
                    x_min:                x,
                    x_max:                x,
                    y_min:                0.0,
                    y_max:                0.0,
                    contributes_to_range: false,
                    legend_entries,
                    render: Box::new(move |x_scale, y_scale| {
                        let px   = x_scale.map(x);
                        let dash = dash.stroke_dasharray().unwrap_or("");
                        let elem = geometry::dashed_line(
                            px, y_scale.pixel_min, px, y_scale.pixel_max,
                            &color_owned, width, dash,
                        );
                        (vec![elem], vec![])
                    }),
                })
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// TextAnchor + AnnotationBuilder
// ---------------------------------------------------------------------------

/// SVG text-anchor alignment for [`AnnotationBuilder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAnchor {
    #[default]
    Start,
    Middle,
    End,
}

impl TextAnchor {
    fn as_str(self) -> &'static str {
        match self { Self::Start => "start", Self::Middle => "middle", Self::End => "end" }
    }
}

/// Builder for a text annotation at a data-space coordinate.
/// Created by [`Layer::annotation`].
pub struct AnnotationBuilder {
    x:        f64,
    y:        f64,
    text:     String,
    color:    Option<String>,
    size:     u32,
    anchor:   TextAnchor,
    offset_x: f64,
    offset_y: f64,
}

impl AnnotationBuilder {
    /// Text color (hex). Default: `"#333333"`.
    pub fn color(mut self, hex: &str) -> Self { self.color = Some(hex.to_string()); self }
    /// Font size in pixels. Default: `12`.
    pub fn size(mut self, px: u32) -> Self { self.size = px; self }
    /// Text anchor. Default: [`TextAnchor::Start`].
    pub fn anchor(mut self, a: TextAnchor) -> Self { self.anchor = a; self }
    /// Pixel nudge after coordinate mapping. Default: `(4.0, -4.0)`.
    pub fn offset(mut self, dx: f64, dy: f64) -> Self { self.offset_x = dx; self.offset_y = dy; self }

    /// Consume this builder and produce a [`Layer`].
    pub fn into_layer(self) -> Layer {
        Layer {
            prepare: Box::new(move || {
                let color_owned = self.color.clone().unwrap_or_else(|| "#333333".to_string());
                let (x, y, size, offset_x, offset_y) =
                    (self.x, self.y, self.size, self.offset_x, self.offset_y);
                let text_owned  = self.text.clone();
                let anchor_str  = self.anchor.as_str();
                Ok(LayerSpec {
                    kind:                 LayerKind::Annotation,
                    x_axis_type:          AxisType::Numeric,
                    y_axis_type:          AxisType::Numeric,
                    x_min:                x,
                    x_max:                x,
                    y_min:                y,
                    y_max:                y,
                    contributes_to_range: false,
                    legend_entries:       vec![],
                    render: Box::new(move |x_scale, y_scale| {
                        let px   = x_scale.map(x) + offset_x;
                        let py   = y_scale.map(y) + offset_y;
                        let elem = geometry::text(px, py, &text_owned, anchor_str, size, &color_owned, 0.0);
                        (vec![elem], vec![])
                    }),
                })
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// LayeredChart
// ---------------------------------------------------------------------------

/// Combines multiple [`Layer`]s into a single chart with a unified coordinate system.
///
/// # Layer ordering
///
/// Layers are drawn in insertion order — first added is drawn first (behind later layers).
///
/// # Axis range
///
/// The unified axis range is computed from the union of all **data** layer ranges.
/// Reference layers (hline, vline, annotation) do not expand the axis range; they are
/// simply clipped by the plot area `<clipPath>` if they fall outside the data range.
pub struct LayeredChart {
    layers: Vec<Layer>,
    title:  Option<String>,
    theme:  Theme,
    width:  u32,
    height: u32,
}

impl Default for LayeredChart {
    fn default() -> Self { Self::new() }
}

impl LayeredChart {
    /// Create a new, empty `LayeredChart`.
    pub fn new() -> Self {
        Self { layers: Vec::new(), title: None, theme: Theme::Default, width: 800, height: 500 }
    }

    /// Add a layer. First added = drawn first (bottom of the stack).
    pub fn layer(mut self, layer: Layer) -> Self { self.layers.push(layer); self }

    /// Set the chart title.
    pub fn title(mut self, title: &str) -> Self { self.title = Some(title.to_string()); self }

    /// Set the visual theme. Default: [`Theme::Default`].
    pub fn theme(mut self, theme: Theme) -> Self { self.theme = theme; self }

    /// Override canvas width in pixels. Default: `800`.
    pub fn width(mut self, px: u32) -> Self { self.width = px; self }

    /// Override canvas height in pixels. Default: `500`.
    pub fn height(mut self, px: u32) -> Self { self.height = px; self }

    /// Validate all layers, unify axis ranges, render, and return a [`Chart`].
    ///
    /// # Errors
    ///
    /// - [`CharcoalError::NoLayers`] — no layers were added.
    /// - [`CharcoalError::IncompatibleLayers`] — mismatched axis types, or a non-composable
    ///   layer kind (Histogram, Heatmap, BoxPlot are excluded by design).
    /// - Any [`CharcoalError`] propagated from a layer's internal `prepare()`.
    pub fn build(self) -> Result<Chart, CharcoalError> {
        // 1. Require at least one layer.
        if self.layers.is_empty() {
            return Err(CharcoalError::NoLayers);
        }

        // 2. Call prepare() on every layer; fail on the first error.
        let mut specs: Vec<LayerSpec> = Vec::with_capacity(self.layers.len());
        for layer in self.layers {
            specs.push((layer.prepare)()?);
        }

        // 3. Compatibility check.
        check_compatibility(&specs)?;

        // 4. Compute unified data range (data layers only).
        let no_data_layers = !specs.iter().any(|s| s.contributes_to_range);

        let x_min = specs.iter().filter(|s| s.contributes_to_range)
            .map(|s| s.x_min).fold(f64::INFINITY, f64::min);
        let x_max = specs.iter().filter(|s| s.contributes_to_range)
            .map(|s| s.x_max).fold(f64::NEG_INFINITY, f64::max);
        let y_min = specs.iter().filter(|s| s.contributes_to_range)
            .map(|s| s.y_min).fold(f64::INFINITY, f64::min);
        let y_max = specs.iter().filter(|s| s.contributes_to_range)
            .map(|s| s.y_max).fold(f64::NEG_INFINITY, f64::max);

        let (x_min, x_max) = if x_min.is_infinite() || x_max.is_infinite() { (0.0, 1.0) } else { (x_min, x_max) };
        let (y_min, y_max) = if y_min.is_infinite() || y_max.is_infinite() { (0.0, 1.0) } else { (y_min, y_max) };

        // 5. Shared ticks and pixel scales.
        let x_tick_vals = nice_ticks(x_min, x_max, 6);
        let y_tick_vals = nice_ticks(y_min, y_max, 6);

        let canvas = SvgCanvas::new(
            self.width, self.height, Margin::default_chart(), ThemeConfig::from(&self.theme),
        );
        let ox = canvas.plot_origin_x();
        let oy = canvas.plot_origin_y();
        let pw = canvas.plot_width();
        let ph = canvas.plot_height();

        let x_scale = LinearScale::new(
            *x_tick_vals.first().unwrap(),
            *x_tick_vals.last().unwrap(),
            ox, ox + pw,
        );
        // SVG y increases downward: data_min → pixel bottom, data_max → pixel top.
        let y_scale = LinearScale::new(
            *y_tick_vals.first().unwrap(),
            *y_tick_vals.last().unwrap(),
            oy + ph, oy,
        );

        // 6 & 7. Collect legend entries then invoke render closures.
        let mut all_elements: Vec<String> = Vec::new();
        let mut all_warnings: Vec<CharcoalWarning> = Vec::new();
        let mut legend_all:   Vec<(String, String)> = Vec::new();
        let mut seen_labels:  HashSet<String> = HashSet::new();

        if no_data_layers {
            all_warnings.push(CharcoalWarning::NoDataLayers);
        }

        for spec in specs {
            // Collect legend entries (dedup by label; first-seen wins).
            for (label, color) in &spec.legend_entries {
                if seen_labels.insert(label.clone()) {
                    legend_all.push((label.clone(), color.clone()));
                }
            }
            let (elems, warns) = (spec.render)(&x_scale, &y_scale);
            all_elements.extend(elems);
            all_warnings.extend(warns);
        }

        // 8. Axis SVG.
        let theme_cfg = ThemeConfig::from(&self.theme);
        let x_labels  = tick_labels_numeric(&x_tick_vals);
        let y_labels  = tick_labels_numeric(&y_tick_vals);
        let x_ticks   = build_tick_marks(&x_tick_vals, &x_labels, &x_scale);
        let y_ticks   = build_tick_marks(&y_tick_vals, &y_labels, &y_scale);
        let x_axis    = compute_axis(&x_scale, &x_ticks, AxisOrientation::Horizontal,
                                     ox, oy, pw, ph, &theme_cfg);
        let y_axis    = compute_axis(&y_scale, &y_ticks, AxisOrientation::Vertical,
                                     ox, oy, pw, ph, &theme_cfg);

        // 9. Assemble.
        let legend = if legend_all.is_empty() { None } else { Some(legend_all) };
        let title  = self.title.as_deref().unwrap_or("");
        let svg    = canvas.render(all_elements, x_axis, y_axis, title, "", "", legend);

        Ok(Chart {
            svg,
            warnings: all_warnings,
            title:    title.to_string(),
            width:    self.width,
            height:   self.height,
        })
    }
}

// ---------------------------------------------------------------------------
// Compatibility check
// ---------------------------------------------------------------------------

/// Validates that a set of `LayerSpec`s can be composed into a single chart.
///
/// Rules:
/// 1. All data layers must agree on x axis type.
/// 2. All data layers must agree on y axis type.
///
/// Reference layers (HLine, VLine, Annotation) are exempt — they always report
/// `AxisType::Numeric` and `contributes_to_range: false`, so they never trigger
/// an axis-type mismatch.
///
/// Note: Histogram, Heatmap, and BoxPlot are excluded from the `Layer` API entirely
/// (no `Layer::histogram` / `Layer::heatmap` / `Layer::box_plot` constructors exist),
/// so they are rejected before reaching this function.
fn check_compatibility(specs: &[LayerSpec]) -> Result<(), CharcoalError> {
    let data_specs: Vec<&LayerSpec> = specs.iter().filter(|s| s.contributes_to_range).collect();

    if let Some(first) = data_specs.first() {
        for spec in &data_specs[1..] {
            if spec.x_axis_type != first.x_axis_type {
                return Err(CharcoalError::IncompatibleLayers {
                    message: format!(
                        "Cannot combine a {:?} x-axis layer with a {:?} x-axis layer",
                        first.x_axis_type, spec.x_axis_type
                    ),
                });
            }
            if spec.y_axis_type != first.y_axis_type {
                return Err(CharcoalError::IncompatibleLayers {
                    message: format!(
                        "Cannot combine a {:?} y-axis layer with a {:?} y-axis layer",
                        first.y_axis_type, spec.y_axis_type
                    ),
                });
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn make_df(xs: &[f64], ys: &[f64]) -> DataFrame {
        DataFrame::new(vec![Series::new("x", xs), Series::new("y", ys)]).unwrap()
    }

    fn scatter_layer(df: &DataFrame) -> Layer {
        Layer::scatter(df).x("x").y("y").into_layer()
    }

    fn line_layer(df: &DataFrame) -> Layer {
        Layer::line(df).x("x").y("y").into_layer()
    }

    // -----------------------------------------------------------------------
    // Happy-path
    // -----------------------------------------------------------------------

    #[test]
    fn single_scatter_layer_produces_valid_svg() {
        let df = make_df(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let svg = LayeredChart::new().layer(scatter_layer(&df)).build().unwrap().svg().to_string();
        assert!(svg.starts_with("<svg"), "must start with <svg");
        assert!(svg.ends_with("</svg>"), "must end with </svg>");
    }

    #[test]
    fn scatter_and_line_produce_both_mark_types() {
        let df  = make_df(&[1.0, 2.0, 3.0, 4.0], &[2.0, 4.0, 3.0, 5.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(line_layer(&df))
            .build()
            .unwrap()
            .svg()
            .to_string();
        assert!(svg.contains("<circle"), "must contain scatter circles");
        assert!(svg.contains("<path"),   "must contain line paths");
    }

    #[test]
    fn extended_line_range_covered_in_ticks() {
        let s_df = make_df(&[0.0, 5.0], &[0.0, 5.0]);
        let l_df = make_df(&[3.0, 10.0], &[3.0, 10.0]);
        let svg  = LayeredChart::new()
            .layer(scatter_layer(&s_df))
            .layer(line_layer(&l_df))
            .build()
            .unwrap()
            .svg()
            .to_string();
        assert!(svg.contains("10"), "unified axis must cover 10");
    }

    #[test]
    fn legend_entries_from_two_layers_both_present() {
        let df = DataFrame::new(vec![
            Series::new("x",   &[1.0f64, 2.0, 3.0]),
            Series::new("y",   &[1.0f64, 2.0, 3.0]),
            Series::new("cat", &["alpha", "alpha", "beta"]),
        ]).unwrap();
        let svg = LayeredChart::new()
            .layer(Layer::scatter(&df).x("x").y("y").color_by("cat").into_layer())
            .layer(Layer::hline(2.5).label("threshold").color("#FF0000").into_layer())
            .build()
            .unwrap()
            .svg()
            .to_string();
        assert!(svg.contains("alpha"));
        assert!(svg.contains("beta"));
        assert!(svg.contains("threshold"));
    }

    #[test]
    fn title_appears_in_svg() {
        let df  = make_df(&[1.0, 2.0], &[1.0, 2.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .title("My Chart")
            .build()
            .unwrap()
            .svg()
            .to_string();
        assert!(svg.contains("My Chart"));
    }

    #[test]
    fn dark_theme_differs_from_default() {
        let df = make_df(&[1.0], &[1.0]);
        let default = LayeredChart::new().layer(scatter_layer(&df)).theme(Theme::Default).build().unwrap().svg().to_string();
        let dark    = LayeredChart::new().layer(scatter_layer(&df)).theme(Theme::Dark).build().unwrap().svg().to_string();
        assert_ne!(default, dark);
    }

    // -----------------------------------------------------------------------
    // Error paths
    // -----------------------------------------------------------------------

    #[test]
    fn empty_chart_returns_no_layers() {
        assert!(matches!(LayeredChart::new().build().unwrap_err(), CharcoalError::NoLayers));
    }

    #[test]
    fn scatter_and_bar_with_mismatched_x_axis_incompatible() {
        let s_df = make_df(&[1.0, 2.0], &[1.0, 2.0]);
        let b_df = DataFrame::new(vec![
            Series::new("cat", &["a", "b"]),
            Series::new("val", &[10.0f64, 20.0]),
        ]).unwrap();
        let err = LayeredChart::new()
            .layer(scatter_layer(&s_df))
            .layer(Layer::bar(&b_df).x("cat").y("val").into_layer())
            .build()
            .unwrap_err();
        assert!(matches!(err, CharcoalError::IncompatibleLayers { .. }));
    }

    #[test]
    fn invalid_column_propagates_column_not_found() {
        let df  = make_df(&[1.0, 2.0], &[1.0, 2.0]);
        let err = LayeredChart::new()
            .layer(Layer::scatter(&df).x("no_such").y("y").into_layer())
            .build()
            .unwrap_err();
        assert!(matches!(err, CharcoalError::ColumnNotFound { .. }));
    }

    // -----------------------------------------------------------------------
    // Scale unification
    // -----------------------------------------------------------------------

    #[test]
    fn unified_range_covers_both_layers() {
        let s_df = make_df(&[0.0, 5.0], &[0.0, 5.0]);
        let l_df = make_df(&[3.0, 10.0], &[3.0, 10.0]);
        let svg  = LayeredChart::new()
            .layer(scatter_layer(&s_df))
            .layer(line_layer(&l_df))
            .build()
            .unwrap()
            .svg()
            .to_string();
        assert!(svg.contains("10"), "unified axis must include 10");
    }

    #[test]
    fn all_null_data_does_not_panic() {
        let df = DataFrame::new(vec![
            Series::new("x", &[None::<f64>, None::<f64>]),
            Series::new("y", &[None::<f64>, None::<f64>]),
        ]).unwrap();
        let svg = LayeredChart::new().layer(scatter_layer(&df)).build().unwrap().svg().to_string();
        assert!(svg.starts_with("<svg"));
    }

    // -----------------------------------------------------------------------
    // Reference layers
    // -----------------------------------------------------------------------

    #[test]
    fn hline_renders_line_element() {
        let df  = make_df(&[1.0, 10.0], &[0.0, 10.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(Layer::hline(5.0).into_layer())
            .build().unwrap().svg().to_string();
        assert!(svg.contains("<line "), "hline must produce <line>");
    }

    #[test]
    fn vline_renders_line_element() {
        let df  = make_df(&[0.0, 10.0], &[0.0, 10.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(Layer::vline(5.0).into_layer())
            .build().unwrap().svg().to_string();
        assert!(svg.contains("<line "), "vline must produce <line>");
    }

    #[test]
    fn annotation_text_in_svg() {
        let df  = make_df(&[1.0, 10.0], &[1.0, 10.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(Layer::annotation(5.0, 5.0, "peak").into_layer())
            .build().unwrap().svg().to_string();
        assert!(svg.contains("peak"));
    }

    #[test]
    fn hline_dashed_has_stroke_dasharray() {
        let df  = make_df(&[0.0, 10.0], &[0.0, 10.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(Layer::hline(5.0).dash(DashStyle::Dashed).into_layer())
            .build().unwrap().svg().to_string();
        assert!(svg.contains("stroke-dasharray"));
    }

    #[test]
    fn hline_label_and_color_in_legend() {
        let df  = make_df(&[0.0, 10.0], &[0.0, 10.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(Layer::hline(5.0).label("threshold").color("#FF0000").into_layer())
            .build().unwrap().svg().to_string();
        assert!(svg.contains("threshold"));
        assert!(svg.contains("#FF0000"));
    }

    #[test]
    fn reference_only_chart_emits_no_data_layers_warning() {
        let chart = LayeredChart::new()
            .layer(Layer::hline(5.0).into_layer())
            .layer(Layer::vline(3.0).into_layer())
            .build()
            .unwrap();
        assert!(chart.warnings().iter().any(|w| matches!(w, CharcoalWarning::NoDataLayers)));
    }

    #[test]
    fn reference_line_does_not_expand_axis_range() {
        let df  = make_df(&[1.0, 10.0], &[1.0, 5.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(Layer::hline(100.0).into_layer())
            .build().unwrap().svg().to_string();
        assert!(!svg.contains(">100<"), "axis ticks must not include 100");
    }

    #[test]
    fn hline_outside_range_still_emits_line_element() {
        let df  = make_df(&[1.0, 10.0], &[1.0, 5.0]);
        let svg = LayeredChart::new()
            .layer(scatter_layer(&df))
            .layer(Layer::hline(100.0).into_layer())
            .build().unwrap().svg().to_string();
        assert!(svg.contains("<line "), "out-of-range hline must still produce a <line> element");
    }
}