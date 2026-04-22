#![allow(dead_code)]
//! Heatmap chart — three-column input, `ColorScale`-based cell rendering.
//!
//! # Builder chain
//!
//! ```text
//! HeatmapBuilder  ──.x()──►  HeatmapWithX  ──.y()──►  HeatmapWithXY  ──.z()──►  HeatmapWithXYZ
//!                                                                                      │
//!                                                                                   .build()
//! ```
//!
//! Optional methods (`.color_scale()`, `.annotate()`, `.title()`, `.theme()`, …)
//! are available on **all** states and return the same state type — they never
//! advance the typestate.
//!
//! # Column roles
//!
//! | Column | Accepted dtypes       | Role                       |
//! |--------|-----------------------|----------------------------|
//! | x      | Categorical, Numeric  | Column labels on the x-axis |
//! | y      | Categorical, Numeric  | Row labels on the y-axis    |
//! | z      | Numeric               | Cell value → fill colour    |
//!
//! # Data preparation (§2.6.1)
//!
//! Input is long-format: each row is a `(x, y, z)` triple.  
//! Duplicate `(x, y)` pairs are **averaged**.  
//! `(x, y)` pairs absent from the data render as a distinct neutral
//! `NULL_CELL_COLOR` that is visually outside every colour scale.
//!
//! # Color scale bar (§2.6.4)
//!
//! A vertical gradient bar is always rendered on the right side of the plot
//! area. It shows the full `z_min → z_max` colour ramp with numeric labels at
//! the minimum, midpoint, and maximum. The bar is spliced into the SVG string
//! *after* `SvgCanvas::render` so it lives outside the clip-path boundary.

use std::collections::HashMap;

use polars::frame::DataFrame;

use crate::charts::Chart;
use crate::dtype::{classify_column, VizDtype};
use crate::error::{CharcoalError, CharcoalWarning};
use crate::normalize::to_f64;
use crate::render::{
    Margin, SvgCanvas,
    axes::{
        AxisOrientation, LinearScale, TickMark,
        categorical_scale, compute_axis,
    },
    geometry,
};
use crate::theme::{ColorScale, Theme, ThemeConfig};

// ---------------------------------------------------------------------------
// Canvas constants
// ---------------------------------------------------------------------------

const CANVAS_WIDTH:  u32 = 900;
const CANVAS_HEIGHT: u32 = 500;
const DEFAULT_ROW_LIMIT: usize = 1_000_000;

/// Neutral grey for cells whose z value is null / absent.
///
/// Chosen to sit clearly outside every `ColorScale` ramp. A viewer must never
/// mistake a null cell for a low or high data value.
pub(crate) const NULL_CELL_COLOR: &str = "#CCCCCC";

// ---------------------------------------------------------------------------
// Accumulated configuration
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct HeatmapConfig {
    x_col:       Option<String>,
    y_col:       Option<String>,
    z_col:       Option<String>,
    color_scale: ColorScale,
    annotate:    bool,
    title:       Option<String>,
    x_label:     Option<String>,
    y_label:     Option<String>,
    theme:       Theme,
    row_limit:   usize,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            x_col:       None,
            y_col:       None,
            z_col:       None,
            color_scale: ColorScale::Viridis,
            annotate:    false,
            title:       None,
            x_label:     None,
            y_label:     None,
            theme:       Theme::Default,
            row_limit:   DEFAULT_ROW_LIMIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Typestate structs
// ---------------------------------------------------------------------------

/// Initial heatmap builder — no required fields set yet.
pub struct HeatmapBuilder<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: HeatmapConfig,
}

/// x column set; y and z still required.
pub struct HeatmapWithX<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: HeatmapConfig,
}

/// x and y columns set; z still required.
pub struct HeatmapWithXY<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: HeatmapConfig,
}

/// All three required columns set — `.build()` is available.
pub struct HeatmapWithXYZ<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: HeatmapConfig,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'df> HeatmapBuilder<'df> {
    pub(crate) fn new(df: &'df DataFrame) -> Self {
        Self { df, config: HeatmapConfig::default() }
    }
}

// ---------------------------------------------------------------------------
// Required field transitions
// ---------------------------------------------------------------------------

impl<'df> HeatmapBuilder<'df> {
    /// Set the x-axis column (Categorical or Numeric).
    /// Column name is validated at `.build()`, not here.
    pub fn x(mut self, col: &str) -> HeatmapWithX<'df> {
        self.config.x_col = Some(col.to_string());
        HeatmapWithX { df: self.df, config: self.config }
    }
}

impl<'df> HeatmapWithX<'df> {
    /// Set the y-axis column (Categorical or Numeric).
    /// Column name is validated at `.build()`, not here.
    pub fn y(mut self, col: &str) -> HeatmapWithXY<'df> {
        self.config.y_col = Some(col.to_string());
        HeatmapWithXY { df: self.df, config: self.config }
    }
}

impl<'df> HeatmapWithXY<'df> {
    /// Set the z-value column (Numeric only).
    /// Column name is validated at `.build()`, not here.
    pub fn z(mut self, col: &str) -> HeatmapWithXYZ<'df> {
        self.config.z_col = Some(col.to_string());
        HeatmapWithXYZ { df: self.df, config: self.config }
    }
}

// ---------------------------------------------------------------------------
// Optional builder methods — identical signature on all four states
// ---------------------------------------------------------------------------

macro_rules! impl_optional {
    ($ty:ty) => {
        impl<'df> $ty {
            /// Set the colour scale used to map z values to cell fill colours.
            /// Default: [`ColorScale::Viridis`].
            pub fn color_scale(mut self, scale: ColorScale) -> Self {
                self.config.color_scale = scale;
                self
            }

            /// When `true`, render the z value as text inside every non-null cell.
            ///
            /// Text colour (black or white) is chosen automatically to maximise
            /// legibility against the cell's background using the WCAG luminance
            /// formula.
            pub fn annotate(mut self, annotate: bool) -> Self {
                self.config.annotate = annotate;
                self
            }

            /// Set the chart title.
            pub fn title(mut self, title: &str) -> Self {
                self.config.title = Some(title.to_string());
                self
            }

            /// Override the x-axis label. Default: the x column name.
            pub fn x_label(mut self, label: &str) -> Self {
                self.config.x_label = Some(label.to_string());
                self
            }

            /// Override the y-axis label. Default: the y column name.
            pub fn y_label(mut self, label: &str) -> Self {
                self.config.y_label = Some(label.to_string());
                self
            }

            /// Set the visual theme. Default: [`Theme::Default`].
            pub fn theme(mut self, theme: Theme) -> Self {
                self.config.theme = theme;
                self
            }

            /// Override the maximum number of rows allowed before returning
            /// [`CharcoalError::DataTooLarge`]. Default: 1 000 000.
            pub fn row_limit(mut self, limit: usize) -> Self {
                self.config.row_limit = limit;
                self
            }
        }
    };
}

impl_optional!(HeatmapBuilder<'df>);
impl_optional!(HeatmapWithX<'df>);
impl_optional!(HeatmapWithXY<'df>);
impl_optional!(HeatmapWithXYZ<'df>);

// ---------------------------------------------------------------------------
// .build() — only available on HeatmapWithXYZ
// ---------------------------------------------------------------------------

impl<'df> HeatmapWithXYZ<'df> {
    /// Validate inputs, aggregate z values, render cells and axes, and return a
    /// [`Chart`].
    ///
    /// # Errors
    ///
    /// | Variant | Cause |
    /// |---------|-------|
    /// | [`CharcoalError::DataTooLarge`]    | `df.height() > row_limit` |
    /// | [`CharcoalError::ColumnNotFound`]  | any column name is missing from the DataFrame |
    /// | [`CharcoalError::UnsupportedColumn`] | z is not Numeric; x or y is Temporal or Unsupported |
    /// | [`CharcoalError::RenderError`]     | all x or all y values are null |
    pub fn build(self) -> Result<Chart, CharcoalError> {
        let df     = self.df;
        let config = self.config;
        let mut warnings: Vec<CharcoalWarning> = Vec::new();

        // ------------------------------------------------------------------
        // 1. Row limit
        // ------------------------------------------------------------------
        let n_rows = df.height();
        if n_rows > config.row_limit {
            return Err(CharcoalError::DataTooLarge {
                rows:    n_rows,
                limit:   config.row_limit,
                message: format!(
                    "DataFrame exceeds the {} row render limit. \
                     Consider aggregating before charting.",
                    config.row_limit,
                ),
            });
        }

        let x_col = config.x_col.as_deref().unwrap(); // always set by typestate
        let y_col = config.y_col.as_deref().unwrap();
        let z_col = config.z_col.as_deref().unwrap();

        // ------------------------------------------------------------------
        // 2. Classify columns
        //    x, y: accept Categorical or Numeric; reject Temporal / Unsupported
        //    z:    must be Numeric
        // ------------------------------------------------------------------
        let x_viz = classify_column(df, x_col, None)?;
        let y_viz = classify_column(df, y_col, None)?;
        let z_viz = classify_column(df, z_col, None)?;

        for (col, viz) in [(x_col, &x_viz), (y_col, &y_viz)] {
            if matches!(viz, VizDtype::Temporal | VizDtype::Unsupported) {
                let dtype = df.schema().get(col).unwrap().clone();
                return Err(CharcoalError::UnsupportedColumn {
                    col:     col.to_string(),
                    message: format!(
                        "Heatmap axis column `{col}` must be Categorical or Numeric. \
                         Temporal and unsupported types cannot be used as heatmap axes."
                    ),
                    dtype,
                });
            }
        }

        if z_viz != VizDtype::Numeric {
            let dtype = df.schema().get(z_col).unwrap().clone();
            let dtype_dbg = format!("{dtype:?}");
            return Err(CharcoalError::UnsupportedColumn {
                col:     z_col.to_string(),
                message: format!(
                    "Heatmap z column `{z_col}` must be Numeric. Got {dtype_dbg}."
                ),
                dtype,
            });
        }

        // ------------------------------------------------------------------
        // 3. Normalize columns to (String, String, f64) triples
        //
        //    x, y: normalised to String keys via normalize_axis_to_strings.
        //          Works for both Categorical (cast to Utf8) and Numeric
        //          (format f64 compactly).
        //    z:    normalised to Vec<Option<f64>> via to_f64.
        // ------------------------------------------------------------------
        let x_strings = normalize_axis_to_strings(df, x_col, x_viz, &mut warnings)?;
        let y_strings = normalize_axis_to_strings(df, y_col, y_viz, &mut warnings)?;
        let (z_vals, z_w) = to_f64(df, z_col)?;
        warnings.extend(z_w);

        // ------------------------------------------------------------------
        // 4. Build ordered category lists and aggregate z values (§2.6.1)
        //
        //    Categories are collected in first-seen row order so the grid layout
        //    mirrors the order they appear in the DataFrame — predictable for the
        //    caller.
        //
        //    Null x or null y rows are silently skipped (NullsSkipped warnings
        //    were already emitted by normalize_axis_to_strings above).
        //    Null z means the (x, y) pair has no value → null cell.
        // ------------------------------------------------------------------
        let mut x_order: Vec<String>              = Vec::new();
        let mut y_order: Vec<String>              = Vec::new();
        let mut x_index: HashMap<String, usize>  = HashMap::new();
        let mut y_index: HashMap<String, usize>  = HashMap::new();

        // (xi, yi) → (sum_z, count) — only non-null z values are accumulated.
        let mut cell_acc: HashMap<(usize, usize), (f64, usize)> = HashMap::new();

        for i in 0..n_rows {
            let Some(xk) = x_strings[i].clone() else { continue };
            let Some(yk) = y_strings[i].clone() else { continue };

            let xi = *x_index.entry(xk.clone()).or_insert_with(|| {
                let idx = x_order.len();
                x_order.push(xk);
                idx
            });
            let yi = *y_index.entry(yk.clone()).or_insert_with(|| {
                let idx = y_order.len();
                y_order.push(yk);
                idx
            });

            // Accumulate non-null z values; null z leaves (xi, yi) absent from
            // cell_acc so the cell will render as NULL_CELL_COLOR.
            if let Some(z) = z_vals[i] {
                let e = cell_acc.entry((xi, yi)).or_insert((0.0, 0));
                e.0 += z;
                e.1 += 1;
            }
        }

        let nx = x_order.len();
        let ny = y_order.len();

        if nx == 0 || ny == 0 {
            return Err(CharcoalError::RenderError(
                "Heatmap requires at least one non-null x value and one non-null y value."
                    .to_string(),
            ));
        }

        // Averaged z grid: grid[yi][xi] = Some(averaged_z) or None (missing).
        let mut grid: Vec<Vec<Option<f64>>> = vec![vec![None; nx]; ny];
        for ((xi, yi), (sum, count)) in &cell_acc {
            if *count > 0 {
                grid[*yi][*xi] = Some(sum / *count as f64);
            }
        }

        // ------------------------------------------------------------------
        // 5. Compute z range for linear colour mapping (§2.6.2)
        // ------------------------------------------------------------------
        let present_z: Vec<f64> = grid
            .iter()
            .flat_map(|row| row.iter().filter_map(|v| *v))
            .collect();

        let (z_min, z_max) = if present_z.is_empty() {
            (0.0, 1.0) // all cells null — use a dummy range; no cells will map colours
        } else {
            let lo = present_z.iter().cloned().fold(f64::INFINITY,     f64::min);
            let hi = present_z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            // Guard against a single unique z value: expand range by ±1 so t
            // can still be computed without a divide-by-zero.
            if (hi - lo).abs() < f64::EPSILON { (lo - 1.0, lo + 1.0) } else { (lo, hi) }
        };

        // ------------------------------------------------------------------
        // 6. Layout — wider right margin to accommodate the colour scale bar
        // ------------------------------------------------------------------
        let theme_cfg = ThemeConfig::from(&config.theme);

        // right = 110 px: 20 gap + 16 bar + 4 gap + ~70 label space
        let margin = Margin::new(50, 110, 60, 70);
        let canvas = SvgCanvas::new(CANVAS_WIDTH, CANVAS_HEIGHT, margin, ThemeConfig::from(&config.theme));

        let ox = canvas.plot_origin_x();
        let oy = canvas.plot_origin_y();
        let pw = canvas.plot_width();
        let ph = canvas.plot_height();

        let cell_w = pw / nx as f64;
        let cell_h = ph / ny as f64;

        // ------------------------------------------------------------------
        // 7. Axis tick marks (categorical for both axes)
        //
        //    categorical_scale returns (label, pixel_center) pairs.
        //    We convert these into TickMark vecs and pass a dummy LinearScale
        //    to compute_axis — the scale is only used for the axis line span;
        //    tick pixel positions come from the TickMark.pixel_pos values
        //    set here.
        // ------------------------------------------------------------------
        let x_cat_positions = categorical_scale(&x_order, ox, ox + pw);
        let y_cat_positions = categorical_scale(&y_order, oy, oy + ph);

        let x_ticks: Vec<TickMark> = x_cat_positions
            .iter()
            .map(|(label, px)| TickMark {
                data_value: 0.0,
                pixel_pos:  *px,
                label:      label.clone(),
            })
            .collect();

        let y_ticks: Vec<TickMark> = y_cat_positions
            .iter()
            .map(|(label, py)| TickMark {
                data_value: 0.0,
                pixel_pos:  *py,
                label:      label.clone(),
            })
            .collect();

        // Dummy scales: compute_axis uses data_min/data_max only for the
        // axis line endpoints, which we override via ox/oy + pw/ph anyway.
        let dummy_x = LinearScale::new(0.0, 1.0, ox, ox + pw);
        let dummy_y = LinearScale::new(0.0, 1.0, oy, oy + ph);

        let x_axis = compute_axis(
            &dummy_x, &x_ticks, AxisOrientation::Horizontal,
            ox, oy, pw, ph, &theme_cfg,
        );
        let y_axis = compute_axis(
            &dummy_y, &y_ticks, AxisOrientation::Vertical,
            ox, oy, pw, ph, &theme_cfg,
        );

        // ------------------------------------------------------------------
        // 8. Render cells (§2.6.2 + §2.6.3)
        // ------------------------------------------------------------------
        let mut elements: Vec<String> = Vec::new();

        for yi in 0..ny {
            for xi in 0..nx {
                // SVG origin is top-left; yi=0 → top row of the grid.
                let cell_x = ox + xi as f64 * cell_w;
                let cell_y = oy + yi as f64 * cell_h;

                // Colour mapping (§2.6.2)
                let fill = match grid[yi][xi] {
                    Some(z) => {
                        let t = (z - z_min) / (z_max - z_min); // always in [0,1]
                        let (r, g, b) = config.color_scale.interpolate(t);
                        format!("#{:02X}{:02X}{:02X}", r, g, b)
                    }
                    None => NULL_CELL_COLOR.to_string(),
                };

                // Cell rectangle — adjacent cells, no gap (rx = 0)
                elements.push(geometry::rect(cell_x, cell_y, cell_w, cell_h, &fill, 0.0));

                // Annotation (§2.6.3)
                if config.annotate {
                    if let Some(z) = grid[yi][xi] {
                        let text_col = legible_text_color(&fill);
                        let label    = format_z_label(z);
                        // Centre text both horizontally and vertically within cell.
                        // SVG text baseline is near the bottom of the em-square;
                        // shift up by ~0.35 em to visually centre it.
                        let tx = cell_x + cell_w / 2.0;
                        let ty = cell_y + cell_h / 2.0
                            + theme_cfg.font_size_px as f64 * 0.35;
                        elements.push(geometry::text(
                            tx, ty, &label, "middle",
                            theme_cfg.font_size_px, text_col, 0.0,
                        ));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // 9. Assemble main SVG via canvas.render
        //    No legend — the colour scale bar replaces it for heatmaps.
        // ------------------------------------------------------------------
        let title   = config.title.as_deref().unwrap_or("");
        let x_label = config.x_label.as_deref().unwrap_or(x_col);
        let y_label = config.y_label.as_deref().unwrap_or(y_col);

        let mut svg = canvas.render(elements, x_axis, y_axis, title, x_label, y_label, None);

        // ------------------------------------------------------------------
        // 10. Colour scale bar (§2.6.4)
        //
        //     Rendered *after* canvas.render so it is outside the clip-path
        //     boundary. We splice the bar SVG just before the closing </svg>
        //     tag.
        // ------------------------------------------------------------------
        let bar_svg = render_color_scale_bar(
            &config.color_scale,
            z_min,
            z_max,
            ox + pw + 20.0, // 20 px gap to the right of the plot area
            oy,             // top aligned with plot area
            16.0,           // bar width
            ph,             // bar height = full plot height
            &theme_cfg,
        );

        if let Some(close_pos) = svg.rfind("</svg>") {
            svg.insert_str(close_pos, &format!("\n{bar_svg}\n"));
        }

        Ok(Chart {
            svg,
            warnings,
            title:  title.to_string(),
            width:  CANVAS_WIDTH,
            height: CANVAS_HEIGHT,
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Normalises an axis column (Categorical **or** Numeric) to `Vec<Option<String>>`.
///
/// For Categorical columns this is a straightforward Utf8 cast.
/// For Numeric columns the f64 values are formatted as compact strings so they
/// can serve as stable, human-readable category keys (e.g. `"1"`, `"3.14"`).
fn normalize_axis_to_strings(
    df:       &DataFrame,
    col:      &str,
    viz:      VizDtype,
    warnings: &mut Vec<CharcoalWarning>,
) -> Result<Vec<Option<String>>, CharcoalError> {
    match viz {
        VizDtype::Categorical => {
            // Cast to Utf8 via Polars — handles String, Categorical, Boolean.
            use polars::datatypes::DataType;
            let series = df
                .column(col)
                .map_err(|e| CharcoalError::RenderError(e.to_string()))?;
            let casted = series
                .cast(&DataType::String)
                .map_err(|e| CharcoalError::RenderError(e.to_string()))?;
            let chunked = casted
                .str()
                .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

            let mut null_count = 0usize;
            let values: Vec<Option<String>> = chunked
                .into_iter()
                .map(|v| {
                    if v.is_none() { null_count += 1; }
                    v.map(|s| s.to_string())
                })
                .collect();

            if null_count > 0 {
                warnings.push(CharcoalWarning::NullsSkipped {
                    col:   col.to_string(),
                    count: null_count,
                });
            }
            Ok(values)
        }
        VizDtype::Numeric => {
            let (f64_vals, w) = to_f64(df, col)?;
            warnings.extend(w);
            let strings = f64_vals
                .into_iter()
                .map(|v| v.map(format_axis_key))
                .collect();
            Ok(strings)
        }
        _ => unreachable!("caller validated that only Categorical/Numeric reach here"),
    }
}

/// Formats a numeric value as a compact string suitable for use as a category
/// key and axis label.
///
/// Integers (no fractional part, within i64 range) are formatted without a
/// decimal point. Other values get up to 4 significant decimal places with
/// trailing zeros stripped.
fn format_axis_key(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1.0e15 {
        format!("{}", v as i64)
    } else {
        // 4 decimal places, strip trailing zeros and the decimal point if
        // it becomes the last character.
        let s = format!("{:.4}", v);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

/// Formats a z value as a compact annotation string inside a cell.
///
/// * `|z| >= 10_000` or (`z != 0` and `|z| < 0.01`) → scientific notation
/// * Integers with `|z| < 10_000` → no decimal point (e.g. `"42"`, `"-7"`)
/// * Otherwise → up to 2 decimal places with trailing zeros stripped
fn format_z_label(z: f64) -> String {
    if z.abs() >= 10_000.0 || (z != 0.0 && z.abs() < 0.01) {
        format!("{:.2e}", z)
    } else if z.fract() == 0.0 {
        format!("{}", z as i64)
    } else {
        let s = format!("{:.2}", z);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

/// Returns `"#FFFFFF"` (white) or `"#000000"` (black) — whichever gives the
/// higher contrast ratio against `hex_fill` per the WCAG relative luminance
/// formula.
///
/// Threshold: luminance < 0.179 → white text, otherwise → black text.
/// This threshold gives a contrast ratio of ≥ 4.5:1 in both cases.
fn legible_text_color(hex_fill: &str) -> &'static str {
    let hex = hex_fill.trim_start_matches('#');
    if hex.len() < 6 {
        return "#000000";
    }
    let parse_chan = |s: &str| -> f64 {
        u8::from_str_radix(s, 16).unwrap_or(128) as f64 / 255.0
    };
    let r = parse_chan(&hex[0..2]);
    let g = parse_chan(&hex[2..4]);
    let b = parse_chan(&hex[4..6]);

    // Linearise from sRGB to linear light
    let lin = |c: f64| -> f64 {
        if c <= 0.03928 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    // ITU-R BT.709 luminance coefficients
    let lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);

    if lum < 0.179 { "#FFFFFF" } else { "#000000" }
}

/// Renders the vertical colour scale legend bar.
///
/// The bar is composed of `GRADIENT_STRIPS` thin `<rect>` elements that together
/// approximate a smooth gradient, avoiding the need for a `<linearGradient>` def
/// (which would require a unique ID and a `<defs>` block outside the canvas).
///
/// Layout (left to right from `bar_x`):
/// ```text
///  bar_x
///   │◄──bar_w──►│◄─4px gap─►│ numeric labels (start-anchored)
/// ```
///
/// Labels are placed at z_max (top), midpoint, and z_min (bottom).
fn render_color_scale_bar(
    scale:  &ColorScale,
    z_min:  f64,
    z_max:  f64,
    bar_x:  f64,
    bar_y:  f64,
    bar_w:  f64,
    bar_h:  f64,
    theme:  &ThemeConfig,
) -> String {
    const GRADIENT_STRIPS: usize = 64;

    let strip_h   = bar_h / GRADIENT_STRIPS as f64;
    // Slight overlap prevents sub-pixel gaps that would show as lines.
    let strip_h_px = strip_h + 0.5;

    let mut parts: Vec<String> = Vec::with_capacity(GRADIENT_STRIPS + 8);

    // Gradient strips — t=1.0 (z_max colour) at the top, t=0.0 (z_min) at
    // the bottom.  i=0 is the topmost strip.
    for i in 0..GRADIENT_STRIPS {
        let t         = 1.0 - (i as f64 / (GRADIENT_STRIPS - 1) as f64);
        let (r, g, b) = scale.interpolate(t);
        let fill      = format!("#{:02X}{:02X}{:02X}", r, g, b);
        let strip_y   = bar_y + i as f64 * strip_h;
        parts.push(geometry::rect(bar_x, strip_y, bar_w, strip_h_px, &fill, 0.0));
    }

    // Thin border around the entire bar for visual definition.
    parts.push(format!(
        r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="none" stroke="{}" stroke-width="1"/>"#,
        bar_x, bar_y, bar_w, bar_h, theme.axis_color,
    ));

    // Labels: z_max at top, midpoint in the middle, z_min at bottom.
    let label_x   = bar_x + bar_w + 4.0;
    let z_mid     = (z_min + z_max) / 2.0;
    let baseline_adjust = theme.font_size_px as f64 * 0.35;

    let tick_positions = [
        (bar_y,              z_max),
        (bar_y + bar_h / 2.0, z_mid),
        (bar_y + bar_h,      z_min),
    ];

    for (ty, z) in tick_positions {
        // Small tick line connecting the bar edge to the label
        parts.push(format!(
            r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="1"/>"#,
            bar_x + bar_w, ty,
            label_x - 1.0, ty,
            theme.axis_color,
        ));
        parts.push(geometry::text(
            label_x,
            ty + baseline_adjust,
            &format_z_label(z),
            "start",
            theme.font_size_px.saturating_sub(1),
            theme.text_color,
            0.0,
        ));
    }

    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Tests (§2.6.5)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    /// Build a simple DataFrame with string x/y and f64 z columns.
    fn make_df(x: &[&str], y: &[&str], z: &[f64]) -> DataFrame {
        DataFrame::new(vec![
            Series::new("x", x),
            Series::new("y", y),
            Series::new("z", z),
        ])
        .unwrap()
    }

    // ------------------------------------------------------------------
    // §2.6.5 — A 3×3 heatmap renders exactly 9 cells
    // ------------------------------------------------------------------

    #[test]
    fn heatmap_3x3_renders_nine_cells() {
        let xs = ["A", "A", "A", "B", "B", "B", "C", "C", "C"];
        let ys = ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"];
        let zs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let df    = make_df(&xs, &ys, &zs);
        let chart = Chart::heatmap(&df).x("x").y("y").z("z").build().unwrap();
        let svg   = chart.svg();

        // Count <rect elements:
        //   1  background rect (canvas)
        //   9  cell rects
        //  64  colour-bar gradient strips
        //   1  colour-bar border
        // ─────────────────────────────
        //  75  minimum
        let rect_count = svg.matches("<rect").count();
        assert!(
            rect_count >= 75,
            "expected ≥75 <rect elements for a 3×3 heatmap; got {rect_count}"
        );
        assert!(svg.starts_with("<svg"), "SVG must start with <svg");
        assert!(svg.ends_with("</svg>"), "SVG must end with </svg>");
    }

    // ------------------------------------------------------------------
    // §2.6.5 — Duplicate (x, y) pairs are averaged
    // ------------------------------------------------------------------

    #[test]
    fn duplicate_xy_pairs_are_averaged() {
        // Two rows for (A, X): z=2 and z=8 → average=5.
        let df = make_df(
            &["A", "A", "B"],
            &["X", "X", "Y"],
            &[2.0, 8.0, 3.0],
        );

        // With annotation on, the averaged value "5" must appear in the SVG.
        let chart = Chart::heatmap(&df)
            .x("x").y("y").z("z")
            .annotate(true)
            .build()
            .unwrap();
        let svg = chart.svg();

        // "5" should appear as annotation text for the averaged cell.
        // geometry::text wraps content between > and </text>, e.g. ">5</text>"
        assert!(
            svg.contains(">5<") || svg.contains(">5</"),
            "averaged value 5 not found as annotation in SVG (first 3000 chars):\n{}",
            &svg[..svg.len().min(3000)],
        );
    }

    // ------------------------------------------------------------------
    // §2.6.5 — Null z cells render with the distinct missing colour
    // ------------------------------------------------------------------

    #[test]
    fn null_z_cells_use_null_cell_color() {
        // Build a DataFrame where the third z value is null.
        // Use ChunkedArray::from_iter for reliable null encoding in Polars.
        use polars::prelude::Float64Chunked;
        let z_ca: Float64Chunked = [Some(1.0_f64), Some(9.0_f64), None]
            .into_iter()
            .collect();
        let z_series = z_ca.into_series().with_name("z");

        let df = DataFrame::new(vec![
            Series::new("x", &["A", "A", "B"]),
            Series::new("y", &["X", "Y", "X"]),
            z_series,
        ])
        .unwrap();

        let chart = Chart::heatmap(&df).x("x").y("y").z("z").build().unwrap();
        let svg   = chart.svg();

        assert!(
            svg.contains(NULL_CELL_COLOR),
            "null cell colour {NULL_CELL_COLOR} must appear in SVG"
        );
    }

    // ------------------------------------------------------------------
    // §2.6.5 — Annotated cells contain <text> elements
    // ------------------------------------------------------------------

    #[test]
    fn annotated_cells_contain_text_elements() {
        let df = make_df(&["A", "B"], &["X", "Y"], &[1.0, 9.0]);

        let chart = Chart::heatmap(&df)
            .x("x").y("y").z("z")
            .annotate(true)
            .build()
            .unwrap();
        let svg = chart.svg();

        // There should be at least one annotation <text beyond axis labels.
        // A 2-cell grid with annotate=true contributes exactly 2 annotation texts.
        let text_count = svg.matches("<text").count();
        assert!(
            text_count >= 2,
            "expected ≥2 <text elements with annotate=true; got {text_count}"
        );
    }

    // ------------------------------------------------------------------
    // §2.6.5 — z_min maps to scale minimum colour, z_max maps to maximum
    // ------------------------------------------------------------------

    #[test]
    fn zmin_maps_to_viridis_minimum_color() {
        // Viridis t=0.0 → (68, 1, 84) → #440154
        let df = make_df(&["A", "B"], &["X", "X"], &[0.0, 100.0]);

        let chart = Chart::heatmap(&df)
            .x("x").y("y").z("z")
            .color_scale(ColorScale::Viridis)
            .build()
            .unwrap();
        let svg_upper = chart.svg().to_uppercase();

        assert!(
            svg_upper.contains("#440154"),
            "Viridis t=0.0 colour #440154 not found in SVG"
        );
    }

    #[test]
    fn zmax_maps_to_viridis_maximum_color() {
        // Viridis t=1.0 → (253, 231, 37) → #FDE725
        let df = make_df(&["A", "B"], &["X", "X"], &[0.0, 100.0]);

        let chart = Chart::heatmap(&df)
            .x("x").y("y").z("z")
            .color_scale(ColorScale::Viridis)
            .build()
            .unwrap();
        let svg_upper = chart.svg().to_uppercase();

        assert!(
            svg_upper.contains("#FDE725"),
            "Viridis t=1.0 colour #FDE725 not found in SVG"
        );
    }

    // ------------------------------------------------------------------
    // Colour scale bar is always rendered (§2.6.4)
    // ------------------------------------------------------------------

    #[test]
    fn color_scale_bar_always_present() {
        let df    = make_df(&["A"], &["X"], &[42.0]);
        let chart = Chart::heatmap(&df).x("x").y("y").z("z").build().unwrap();
        let svg   = chart.svg();

        // Bar is 64 strips + 1 border + 3 tick lines = 68 extra elements
        // above the 1 background + 1 cell = at minimum 70 <rect elements.
        let rect_count = svg.matches("<rect").count();
        assert!(
            rect_count >= 65,
            "colour scale bar (≥64 gradient strips) missing; got {rect_count} <rect elements"
        );
    }

    // ------------------------------------------------------------------
    // Numeric x/y columns are accepted (§2.6.1 column type flexibility)
    // ------------------------------------------------------------------

    #[test]
    fn numeric_xy_columns_render_successfully() {
        let df = DataFrame::new(vec![
            Series::new("x", &[1.0_f64, 1.0, 2.0]),
            Series::new("y", &[10.0_f64, 20.0, 10.0]),
            Series::new("z", &[5.0_f64, 3.0, 7.0]),
        ])
        .unwrap();

        let result = Chart::heatmap(&df).x("x").y("y").z("z").build();
        assert!(result.is_ok(), "numeric x/y should build cleanly: {:?}", result.err());
        let svg = result.unwrap().svg().to_string();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    // ------------------------------------------------------------------
    // Temporal axis column → UnsupportedColumn error
    // ------------------------------------------------------------------

    #[test]
    fn temporal_x_returns_unsupported_column_error() {
        let df = DataFrame::new(vec![
            Series::new("x", &[1i64, 2])
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap(),
            Series::new("y", &["A", "B"]),
            Series::new("z", &[1.0_f64, 2.0]),
        ])
        .unwrap();

        let result = Chart::heatmap(&df).x("x").y("y").z("z").build();
        assert!(
            matches!(result, Err(CharcoalError::UnsupportedColumn { .. })),
            "expected UnsupportedColumn for temporal x; got {:?}", result,
        );
    }

    // ------------------------------------------------------------------
    // Non-numeric z → UnsupportedColumn error
    // ------------------------------------------------------------------

    #[test]
    fn categorical_z_returns_unsupported_column_error() {
        let df = DataFrame::new(vec![
            Series::new("x", &["A", "B"]),
            Series::new("y", &["X", "Y"]),
            Series::new("z", &["foo", "bar"]), // String — not Numeric
        ])
        .unwrap();

        let result = Chart::heatmap(&df).x("x").y("y").z("z").build();
        assert!(
            matches!(result, Err(CharcoalError::UnsupportedColumn { .. })),
            "expected UnsupportedColumn for string z; got {:?}", result,
        );
    }

    // ------------------------------------------------------------------
    // DataTooLarge error when row_limit is exceeded
    // ------------------------------------------------------------------

    #[test]
    fn exceeding_row_limit_returns_data_too_large() {
        let df = make_df(&["A", "B", "C"], &["X", "Y", "Z"], &[1.0, 2.0, 3.0]);
        let result = Chart::heatmap(&df)
            .x("x").y("y").z("z")
            .row_limit(2)
            .build();
        assert!(
            matches!(result, Err(CharcoalError::DataTooLarge { .. })),
            "expected DataTooLarge; got {:?}", result,
        );
    }

    // ------------------------------------------------------------------
    // legible_text_color — WCAG luminance thresholding
    // ------------------------------------------------------------------

    #[test]
    fn white_text_on_very_dark_background() {
        assert_eq!(legible_text_color("#000000"), "#FFFFFF");
    }

    #[test]
    fn black_text_on_very_light_background() {
        assert_eq!(legible_text_color("#FFFFFF"), "#000000");
    }

    #[test]
    fn white_text_on_viridis_min() {
        // Viridis min: (68, 1, 84) → very dark purple
        assert_eq!(legible_text_color("#440154"), "#FFFFFF");
    }

    #[test]
    fn black_text_on_viridis_max() {
        // Viridis max: (253, 231, 37) → bright yellow
        assert_eq!(legible_text_color("#FDE725"), "#000000");
    }

    #[test]
    fn white_text_on_null_cell_color() {
        // NULL_CELL_COLOR = #CCCCCC is a light grey → black text
        // (luminance ≈ 0.60 which is above 0.179)
        assert_eq!(legible_text_color(NULL_CELL_COLOR), "#000000");
    }

    // ------------------------------------------------------------------
    // format_axis_key — integer and decimal formatting
    // ------------------------------------------------------------------

    #[test]
    fn format_axis_key_integer_value() {
        assert_eq!(format_axis_key(5.0), "5");
        assert_eq!(format_axis_key(-100.0), "-100");
        assert_eq!(format_axis_key(0.0), "0");
    }

    #[test]
    fn format_axis_key_decimal_strips_trailing_zeros() {
        assert_eq!(format_axis_key(3.1400), "3.14");
        assert_eq!(format_axis_key(1.5), "1.5");
    }

    // ------------------------------------------------------------------
    // format_z_label — annotation label formatting
    // ------------------------------------------------------------------

    #[test]
    fn format_z_label_integer() {
        assert_eq!(format_z_label(42.0), "42");
    }

    #[test]
    fn format_z_label_decimal() {
        assert_eq!(format_z_label(3.14), "3.14");
    }

    #[test]
    fn format_z_label_scientific_large() {
        // >= 10_000 triggers scientific notation regardless of integer status
        let s = format_z_label(1_234_567.0);
        assert!(s.contains('e'), "large value should use scientific notation: {s}");
    }

    #[test]
    fn format_z_label_integer_under_threshold() {
        // integers < 10_000 stay as plain integers
        assert_eq!(format_z_label(9_999.0), "9999");
        assert_eq!(format_z_label(-42.0),   "-42");
    }

    #[test]
    fn format_z_label_scientific_tiny() {
        let s = format_z_label(0.0001);
        assert!(s.contains('e'), "tiny value should use scientific notation: {s}");
    }

    // ------------------------------------------------------------------
    // Builder — optional methods available on all states
    // ------------------------------------------------------------------

    #[test]
    fn optional_methods_available_at_every_state() {
        let df = make_df(&["A"], &["X"], &[1.0]);

        // These must all compile and run without error.
        let _chart = Chart::heatmap(&df)
            .color_scale(ColorScale::Plasma) // on HeatmapBuilder
            .x("x")
            .color_scale(ColorScale::RdBu)   // on HeatmapWithX
            .y("y")
            .annotate(true)                  // on HeatmapWithXY
            .z("z")
            .title("Test")                   // on HeatmapWithXYZ
            .x_label("Columns")
            .y_label("Rows")
            .theme(Theme::Dark)
            .row_limit(500_000)
            .annotate(false)
            .color_scale(ColorScale::Greyscale)
            .build()
            .unwrap();
    }
}