//! Box plot builder (`Chart::box_plot(&df)`).
//!
//! Follows the same typestate pattern as `scatter.rs`, `bar.rs`, and `line.rs`.
//!
//! # Column roles
//!
//! | Column | Required dtype | Role |
//! |--------|---------------|------|
//! | `x`    | Categorical   | Group labels (one box per unique value) |
//! | `y`    | Numeric       | Distribution of values being summarised |
//!
//! # Statistics
//!
//! For each group the following five-number summary is computed from non-null y
//! values using a sort-based linear-interpolation percentile:
//!
//! - **minimum** — non-outlier lower fence (≥ Q1 − 1.5 × IQR)
//! - **Q1** — 25th percentile
//! - **median** — 50th percentile
//! - **Q3** — 75th percentile
//! - **maximum** — non-outlier upper fence (≤ Q3 + 1.5 × IQR)
//! - **outliers** — individual values outside the fences
//!
//! # Rendering
//!
//! Each group renders as:
//! 1. A vertical rectangle from Q1 → Q3 (the IQR box).
//! 2. A horizontal line at the median.
//! 3. Vertical whiskers from the box edges to the non-outlier min/max.
//! 4. Individual point markers for outliers (or all points when `PointDisplay::All`).
//!
//! # Options
//!
//! - `.notched(bool)` — notch the box at the median to show a 95% CI.
//! - `.points(PointDisplay)` — control which individual data points are shown.

use polars::frame::DataFrame;

use crate::charts::Chart;
use crate::dtype::{classify_column, VizDtype};
use crate::error::{CharcoalError, CharcoalWarning};
use crate::normalize::{to_f64, to_string};
use crate::render::{
    SvgCanvas, Margin,
    axes::{
        AxisOrientation, LinearScale, TickMark,
        build_tick_marks, compute_axis,
        nice_ticks, tick_labels_numeric, categorical_scale,
    },
    geometry,
};
use crate::theme::{Theme, ThemeConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CANVAS_WIDTH:       u32   = 800;
const CANVAS_HEIGHT:      u32   = 500;
const DEFAULT_ROW_LIMIT:  usize = 1_000_000;
/// Fractional width of the IQR box relative to the category band.
const BOX_WIDTH_FRAC:     f64   = 0.5;
/// Fractional width of the whisker cap relative to the box.
const WHISKER_CAP_FRAC:   f64   = 0.4;
/// Radius of individual data-point markers (pixels).
const POINT_RADIUS:       f64   = 3.0;
/// Maximum absolute jitter offset for `PointDisplay::All` (pixels).
const JITTER_MAX_PX:      f64   = 6.0;

// ---------------------------------------------------------------------------
// Public enums
// ---------------------------------------------------------------------------

/// Controls which individual data points are rendered on top of each box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointDisplay {
    /// Only outliers are shown as individual points (default).
    Outliers,
    /// All individual data points are shown, with a deterministic x-jitter to
    /// reduce overplotting.
    All,
    /// No individual points are shown.
    None,
}

impl Default for PointDisplay {
    fn default() -> Self {
        Self::Outliers
    }
}

// ---------------------------------------------------------------------------
// Accumulated configuration
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct BoxPlotConfig {
    pub x_col:      Option<String>,
    pub y_col:      Option<String>,
    pub title:      Option<String>,
    pub x_label:    Option<String>,
    pub y_label:    Option<String>,
    pub theme:      Theme,
    pub notched:    bool,
    pub points:     PointDisplay,
    pub row_limit:  usize,
}

impl Default for BoxPlotConfig {
    fn default() -> Self {
        Self {
            x_col:     None,
            y_col:     None,
            title:     None,
            x_label:   None,
            y_label:   None,
            theme:     Theme::Default,
            notched:   false,
            points:    PointDisplay::Outliers,
            row_limit: DEFAULT_ROW_LIMIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder states
// ---------------------------------------------------------------------------

/// Initial box-plot builder — no required fields set yet.
pub struct BoxPlotBuilder<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: BoxPlotConfig,
}

/// x-axis column set; y still required.
pub struct BoxPlotWithX<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: BoxPlotConfig,
}

/// Both required fields set — `.build()` is available.
pub struct BoxPlotWithXY<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: BoxPlotConfig,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'df> BoxPlotBuilder<'df> {
    pub(crate) fn new(df: &'df DataFrame) -> Self {
        Self { df, config: BoxPlotConfig::default() }
    }
}

// ---------------------------------------------------------------------------
// Optional setter methods (macro-stamped onto all three states)
// ---------------------------------------------------------------------------

macro_rules! impl_box_plot_optional_setters {
    ($t:ty) => {
        impl<'df> $t {
            /// Chart title rendered above the plot area.
            pub fn title(mut self, title: &str) -> Self {
                self.config.title = Some(title.to_string());
                self
            }

            /// Label rendered below the x-axis.
            pub fn x_label(mut self, label: &str) -> Self {
                self.config.x_label = Some(label.to_string());
                self
            }

            /// Label rendered beside the y-axis.
            pub fn y_label(mut self, label: &str) -> Self {
                self.config.y_label = Some(label.to_string());
                self
            }

            /// Visual theme. Defaults to [`Theme::Default`].
            pub fn theme(mut self, theme: Theme) -> Self {
                self.config.theme = theme;
                self
            }

            /// When `true`, notch the IQR box at the median to show a 95% confidence
            /// interval. The notch half-width is `1.58 × IQR / √N`.
            ///
            /// If the notch extends beyond Q1 or Q3 (possible with very small N) it is
            /// clamped and a [`CharcoalWarning`] is emitted.
            pub fn notched(mut self, notched: bool) -> Self {
                self.config.notched = notched;
                self
            }

            /// Controls which individual data points are rendered.
            ///
            /// - [`PointDisplay::Outliers`] (default) — only outlier points.
            /// - [`PointDisplay::All`] — all points with a small deterministic x-jitter.
            /// - [`PointDisplay::None`] — no individual points.
            pub fn points(mut self, points: PointDisplay) -> Self {
                self.config.points = points;
                self
            }

            /// Row limit above which `.build()` returns [`CharcoalError::DataTooLarge`].
            /// Defaults to 1,000,000.
            pub fn row_limit(mut self, limit: usize) -> Self {
                self.config.row_limit = limit;
                self
            }
        }
    };
}

impl_box_plot_optional_setters!(BoxPlotBuilder<'df>);
impl_box_plot_optional_setters!(BoxPlotWithX<'df>);
impl_box_plot_optional_setters!(BoxPlotWithXY<'df>);

// ---------------------------------------------------------------------------
// Required field transitions
// ---------------------------------------------------------------------------

impl<'df> BoxPlotBuilder<'df> {
    /// Set the categorical x-axis column (group labels).
    pub fn x(mut self, col: &str) -> BoxPlotWithX<'df> {
        self.config.x_col = Some(col.to_string());
        BoxPlotWithX { df: self.df, config: self.config }
    }
}

impl<'df> BoxPlotWithX<'df> {
    /// Set the numeric y-axis column (values). Unlocks `.build()`.
    pub fn y(mut self, col: &str) -> BoxPlotWithXY<'df> {
        self.config.y_col = Some(col.to_string());
        BoxPlotWithXY { df: self.df, config: self.config }
    }
}

// ---------------------------------------------------------------------------
// .build()
// ---------------------------------------------------------------------------

impl<'df> BoxPlotWithXY<'df> {
    /// Validate columns, compute per-group statistics, render boxes, and return a
    /// [`Chart`].
    ///
    /// # Errors
    ///
    /// - [`CharcoalError::DataTooLarge`] if `df.height() > row_limit`.
    /// - [`CharcoalError::ColumnNotFound`] for any missing column name.
    /// - [`CharcoalError::UnsupportedColumn`] if x is not Categorical or y is not Numeric.
    pub fn build(self) -> Result<Chart, CharcoalError> {
        let df     = self.df;
        let config = self.config;
        let mut warnings: Vec<CharcoalWarning> = Vec::new();

        // ------------------------------------------------------------------
        // 1. Row limit check
        // ------------------------------------------------------------------
        let n_rows = df.height();
        if n_rows > config.row_limit {
            return Err(CharcoalError::DataTooLarge {
                rows:    n_rows,
                limit:   config.row_limit,
                message: format!(
                    "DataFrame exceeds the {} row render limit. \
                     Consider df.sample({}) or an aggregation before charting.",
                    config.row_limit,
                    config.row_limit / 2,
                ),
            });
        }

        // ------------------------------------------------------------------
        // 2. Validate and normalize x (must be Categorical)
        // ------------------------------------------------------------------
        let x_col = config.x_col.as_deref().unwrap(); // typestate guarantees presence
        let x_viz = classify_column(df, x_col, None)?;

        if x_viz == VizDtype::Numeric || x_viz == VizDtype::Temporal {
            let dtype = df.schema().get(x_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     x_col.to_string(),
                dtype,
                message: format!(
                    "The x column of a box plot must be Categorical (String or Boolean), \
                     not {:?}. Each unique x value defines one box.",
                    df.schema().get(x_col).unwrap()
                ),
            });
        }
        if x_viz == VizDtype::Unsupported {
            let dtype = df.schema().get(x_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     x_col.to_string(),
                dtype,
                message: "The x column of a box plot must be Categorical (String or Boolean)."
                    .to_string(),
            });
        }

        // Normalize x: nulls become "null"
        let (x_raw, x_warnings) = to_string(df, x_col)?;
        warnings.extend(x_warnings);
        let x_vals: Vec<String> = x_raw
            .into_iter()
            .map(|v| v.unwrap_or_else(|| "null".to_string()))
            .collect();

        // ------------------------------------------------------------------
        // 3. Validate and normalize y (must be Numeric)
        // ------------------------------------------------------------------
        let y_col = config.y_col.as_deref().unwrap();
        let y_viz = classify_column(df, y_col, None)?;

        if y_viz != VizDtype::Numeric {
            let dtype = df.schema().get(y_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     y_col.to_string(),
                dtype,
                message: "The y column of a box plot must be Numeric.".to_string(),
            });
        }

        let (y_vals_raw, _) = to_f64(df, y_col)?;
        // Null y values are excluded from all per-group computations (emit warning).
        let mut null_y_count = 0usize;
        let y_vals: Vec<Option<f64>> = y_vals_raw
            .into_iter()
            .inspect(|v| { if v.is_none() { null_y_count += 1; } })
            .collect();
        if null_y_count > 0 {
            warnings.push(CharcoalWarning::NullsSkipped {
                col:   y_col.to_string(),
                count: null_y_count,
            });
        }

        // ------------------------------------------------------------------
        // 4. Collect category order (first-seen) and build per-group value lists
        // ------------------------------------------------------------------
        let mut categories: Vec<String> = Vec::new();
        for v in &x_vals {
            if !categories.contains(v) {
                categories.push(v.clone());
            }
        }

        // For each category, collect the non-null y values.
        let mut group_values: Vec<Vec<f64>> = vec![Vec::new(); categories.len()];
        for row in 0..n_rows {
            if let Some(y) = y_vals[row] {
                let ci = categories.iter().position(|c| c == &x_vals[row]).unwrap();
                group_values[ci].push(y);
            }
        }

        // ------------------------------------------------------------------
        // 5. Compute per-group statistics
        // ------------------------------------------------------------------
        let mut group_stats: Vec<Option<GroupStats>> = Vec::with_capacity(categories.len());
        for (ci, vals) in group_values.iter().enumerate() {
            if vals.is_empty() {
                group_stats.push(None);
                continue;
            }
            let mut stats = compute_stats(vals);
            // Notch calculation
            if config.notched {
                let n = vals.len() as f64;
                let iqr = stats.q3 - stats.q1;
                let half_notch = 1.58 * iqr / n.sqrt();
                let mut notch_lo = stats.median - half_notch;
                let mut notch_hi = stats.median + half_notch;
                let mut clamped = false;
                if notch_lo < stats.q1 {
                    notch_lo = stats.q1;
                    clamped = true;
                }
                if notch_hi > stats.q3 {
                    notch_hi = stats.q3;
                    clamped = true;
                }
                if clamped {
                    warnings.push(CharcoalWarning::NotchClamped {
                        group: categories[ci].clone(),
                    });
                }
                stats.notch_lo = Some(notch_lo);
                stats.notch_hi = Some(notch_hi);
            }
            group_stats.push(Some(stats));
        }

        // ------------------------------------------------------------------
        // 6. Compute y-axis range across all groups
        // ------------------------------------------------------------------
        let all_y: Vec<f64> = group_values.iter().flatten().copied().collect();
        let (y_data_min, y_data_max) = if all_y.is_empty() {
            (0.0_f64, 1.0_f64)
        } else {
            let lo = all_y.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (lo, hi)
        };

        let y_tick_vals = nice_ticks(y_data_min, y_data_max, 6);

        // ------------------------------------------------------------------
        // 7. Canvas and scales
        // ------------------------------------------------------------------
        let theme_cfg = ThemeConfig::from(&config.theme);
        let canvas = SvgCanvas::new(
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
            Margin::default_chart(),
            ThemeConfig::from(&config.theme),
        );
        let ox = canvas.plot_origin_x();
        let oy = canvas.plot_origin_y();
        let pw = canvas.plot_width();
        let ph = canvas.plot_height();

        let cat_positions = categorical_scale(&categories, 0.0, 1.0);
        let band_px = pw / categories.len() as f64;

        let y_scale = LinearScale::new(
            *y_tick_vals.first().unwrap(),
            *y_tick_vals.last().unwrap(),
            oy + ph,   // data_min → pixel bottom
            oy,        // data_max → pixel top
        );

        let fill_color   = theme_cfg.palette[0];
        let stroke_color = theme_cfg.axis_color;

        // ------------------------------------------------------------------
        // 8. Render boxes
        // ------------------------------------------------------------------
        let mut elements: Vec<String> = Vec::new();

        for (ci, (_cat_label, cat_norm)) in cat_positions.iter().enumerate() {
            let center_px = ox + cat_norm * pw;
            let box_half  = band_px * BOX_WIDTH_FRAC / 2.0;
            let cap_half  = box_half * WHISKER_CAP_FRAC;

            let stats = match &group_stats[ci] {
                None    => continue,
                Some(s) => s,
            };

            let py_q1     = y_scale.map(stats.q1);
            let py_q3     = y_scale.map(stats.q3);
            let py_med    = y_scale.map(stats.median);
            let py_whi_lo = y_scale.map(stats.whisker_lo);
            let py_whi_hi = y_scale.map(stats.whisker_hi);

            // ── IQR box ────────────────────────────────────────────────────
            // SVG y increases downward, so Q3 pixel < Q1 pixel (Q3 is higher).
            let box_top = py_q3.min(py_q1);
            let box_h   = (py_q1 - py_q3).abs().max(0.5); // at least 0.5px for zero-IQR

            if config.notched {
                // Notched box: rendered as a polygon with a pinch at the median.
                let (nl, nh) = (
                    y_scale.map(stats.notch_lo.unwrap_or(stats.median)),
                    y_scale.map(stats.notch_hi.unwrap_or(stats.median)),
                );
                // Polygon vertices (clockwise from top-left):
                // top-left → top-right, narrow at notch-hi (upper notch), expand at median,
                // narrow at notch-lo (lower notch), bottom-right → bottom-left
                let box_left  = center_px - box_half;
                let box_right = center_px + box_half;
                let notch_indent = box_half * 0.35; // indentation width
                let nl_px = nh.min(py_med); // nh is the upper notch (smaller y = higher)
                let nh_px = nl.max(py_med); // nl is the lower notch (larger y = lower)

                let poly_pts = vec![
                    (box_left,              box_top),     // top-left
                    (box_right,             box_top),     // top-right
                    (box_right,             nl_px),       // right side down to upper notch
                    (center_px + notch_indent, py_med),   // right notch point
                    (box_right,             nh_px),       // right side below median
                    (box_right,             box_top + box_h), // bottom-right
                    (box_left,              box_top + box_h), // bottom-left
                    (box_left,              nh_px),       // left side below median
                    (center_px - notch_indent, py_med),   // left notch point
                    (box_left,              nl_px),       // left side upper notch
                ];
                elements.push(geometry::polygon(&poly_pts, fill_color, stroke_color, 0.7));
            } else {
                // Plain rectangular box
                elements.push(geometry::rect(
                    center_px - box_half, box_top,
                    box_half * 2.0,       box_h,
                    fill_color, 0.0,
                ));
                // Box border (stroke)
                elements.push(format!(
                    r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="none" stroke="{}" stroke-width="1.50"/>"#,
                    center_px - box_half, box_top,
                    box_half * 2.0,       box_h,
                    stroke_color,
                ));
            }

            // ── Median line ────────────────────────────────────────────────
            elements.push(geometry::line(
                center_px - box_half, py_med,
                center_px + box_half, py_med,
                stroke_color, 2.0,
            ));

            // ── Upper whisker (Q3 → non-outlier max) ──────────────────────
            // SVG: lower y = higher on screen, so whisker_hi maps to smaller py
            elements.push(geometry::line(
                center_px, py_q3,
                center_px, py_whi_hi,
                stroke_color, 1.5,
            ));
            // upper whisker cap
            elements.push(geometry::line(
                center_px - cap_half, py_whi_hi,
                center_px + cap_half, py_whi_hi,
                stroke_color, 1.5,
            ));

            // ── Lower whisker (Q1 → non-outlier min) ──────────────────────
            elements.push(geometry::line(
                center_px, py_q1,
                center_px, py_whi_lo,
                stroke_color, 1.5,
            ));
            // lower whisker cap
            elements.push(geometry::line(
                center_px - cap_half, py_whi_lo,
                center_px + cap_half, py_whi_lo,
                stroke_color, 1.5,
            ));

            // ── Points ─────────────────────────────────────────────────────
            match config.points {
                PointDisplay::None => {}
                PointDisplay::Outliers => {
                    for &ov in &stats.outliers {
                        let py = y_scale.map(ov);
                        elements.push(geometry::circle(
                            center_px, py, POINT_RADIUS, stroke_color, 0.8,
                        ));
                    }
                }
                PointDisplay::All => {
                    let jitter_vals = &group_values[ci];
                    for (vi, &yv) in jitter_vals.iter().enumerate() {
                        let jitter = deterministic_jitter(ci, vi, box_half);
                        let py = y_scale.map(yv);
                        elements.push(geometry::circle(
                            center_px + jitter, py, POINT_RADIUS, stroke_color, 0.6,
                        ));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // 9. Axes
        // ------------------------------------------------------------------
        let y_labels  = tick_labels_numeric(&y_tick_vals);
        let val_ticks = build_tick_marks(&y_tick_vals, &y_labels, &y_scale);

        let cat_tick_marks: Vec<TickMark> = cat_positions
            .iter()
            .map(|(cat_label, norm)| TickMark {
                data_value: *norm,
                pixel_pos:  ox + norm * pw,
                label:      cat_label.clone(),
            })
            .collect();

        let cat_scale = LinearScale::new(0.0, 1.0, 0.0, 1.0);

        let x_axis = compute_axis(
            &cat_scale, &cat_tick_marks, AxisOrientation::Horizontal,
            ox, oy, pw, ph, &theme_cfg,
        );
        let y_axis = compute_axis(
            &y_scale, &val_ticks, AxisOrientation::Vertical,
            ox, oy, pw, ph, &theme_cfg,
        );

        // ------------------------------------------------------------------
        // 10. Assemble Chart
        // ------------------------------------------------------------------
        let title   = config.title.as_deref().unwrap_or("");
        let x_label = config.x_label.as_deref().unwrap_or(x_col);
        let y_label = config.y_label.as_deref().unwrap_or(y_col);

        let svg = canvas.render(elements, x_axis, y_axis, title, x_label, y_label, None);

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
// Per-group statistics
// ---------------------------------------------------------------------------

/// Five-number summary plus outliers and optional notch bounds for one group.
#[derive(Debug, Clone)]
struct GroupStats {
    /// Non-outlier minimum (lower whisker end).
    whisker_lo: f64,
    q1:         f64,
    median:     f64,
    q3:         f64,
    /// Non-outlier maximum (upper whisker end).
    whisker_hi: f64,
    /// Values outside the 1.5 × IQR fences.
    outliers:   Vec<f64>,
    /// Lower notch bound (set only when `.notched(true)`).
    notch_lo:   Option<f64>,
    /// Upper notch bound (set only when `.notched(true)`).
    notch_hi:   Option<f64>,
}

/// Compute a [`GroupStats`] from a non-empty slice of values (nulls already removed).
///
/// Percentile implementation: sort the data, then use linear interpolation at
/// index `floor((N − 1) × p)` for each percentile `p`.
fn compute_stats(vals: &[f64]) -> GroupStats {
    debug_assert!(!vals.is_empty());

    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1     = percentile(&sorted, 0.25);
    let median = percentile(&sorted, 0.50);
    let q3     = percentile(&sorted, 0.75);
    let iqr    = q3 - q1;

    let lo_fence = q1 - 1.5 * iqr;
    let hi_fence = q3 + 1.5 * iqr;

    let mut outliers: Vec<f64> = Vec::new();
    let mut whisker_lo = q1; // will be updated to lowest non-outlier
    let mut whisker_hi = q3; // will be updated to highest non-outlier

    for &v in &sorted {
        if v < lo_fence || v > hi_fence {
            outliers.push(v);
        } else {
            if v < whisker_lo { whisker_lo = v; }
            if v > whisker_hi { whisker_hi = v; }
        }
    }

    GroupStats {
        whisker_lo,
        q1,
        median,
        q3,
        whisker_hi,
        outliers,
        notch_lo: None,
        notch_hi: None,
    }
}

/// Linear-interpolation percentile for a **sorted** non-empty slice.
///
/// For N values and percentile p, the continuous index is `i = (N − 1) × p`.
/// The result is `sorted[floor(i)] + frac(i) × (sorted[ceil(i)] − sorted[floor(i)])`.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    debug_assert!(n > 0);
    if n == 1 {
        return sorted[0];
    }
    let idx = (n - 1) as f64 * p;
    let lo  = idx.floor() as usize;
    let hi  = idx.ceil()  as usize;
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ---------------------------------------------------------------------------
// Deterministic jitter
// ---------------------------------------------------------------------------

/// Returns a pixel x-offset for `PointDisplay::All` to reduce overplotting.
///
/// Uses a simple LCG seeded by `(category_index, value_index)` so the jitter is
/// fully deterministic across renders — no `rand` crate dependency required.
fn deterministic_jitter(cat_idx: usize, val_idx: usize, box_half: f64) -> f64 {
    // LCG parameters (Numerical Recipes)
    let seed = (cat_idx as u64).wrapping_mul(6364136223846793005)
        .wrapping_add(val_idx as u64)
        .wrapping_add(1442695040888963407);
    let lcg = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    // Map to [−JITTER_MAX_PX, +JITTER_MAX_PX], clamped to box half-width.
    let t = (lcg as f64) / (u64::MAX as f64); // [0, 1)
    let max = JITTER_MAX_PX.min(box_half * 0.8);
    (t - 0.5) * 2.0 * max
}

// ---------------------------------------------------------------------------
// CharcoalWarning extension — NotchClamped variant
//
// NOTE: Add this variant to `error.rs`:
//
//   NotchClamped { group: String },
//
// And its Display arm:
//
//   Self::NotchClamped { group } => write!(
//       f,
//       "Notch bounds for group \"{group}\" extended beyond Q1/Q3 and were clamped. \
//        Consider using a larger sample or disabling notches for this group.",
//   ),
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a minimal DataFrame with one categorical and one numeric column.
    fn make_df(cats: &[&str], vals: &[f64]) -> DataFrame {
        DataFrame::new(vec![
            Series::new("group", cats),
            Series::new("value", vals),
        ])
        .unwrap()
    }

    /// Build a DataFrame with optional (nullable) y values.
    fn make_df_opt(cats: &[&str], vals: &[Option<f64>]) -> DataFrame {
        DataFrame::new(vec![
            Series::new("group", cats),
            Series::new("value", vals),
        ])
        .unwrap()
    }

    // ── percentile / compute_stats ────────────────────────────────────────────

    #[test]
    fn percentile_single_value_returns_that_value() {
        assert!((percentile(&[42.0], 0.25) - 42.0).abs() < 1e-9);
        assert!((percentile(&[42.0], 0.50) - 42.0).abs() < 1e-9);
        assert!((percentile(&[42.0], 0.75) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_two_values_median_is_midpoint() {
        let s = vec![0.0, 10.0];
        assert!((percentile(&s, 0.50) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn five_number_summary_known_dataset() {
        // Dataset: [2, 4, 4, 4, 5, 5, 7, 9]  (N=8)
        // Sorted: [2, 4, 4, 4, 5, 5, 7, 9]
        // Q1 index = 7*0.25 = 1.75  → 4 + 0.75*(4-4) = 4.0
        // Median index = 7*0.50 = 3.5 → 4 + 0.5*(5-4) = 4.5
        // Q3 index = 7*0.75 = 5.25 → 5 + 0.25*(7-5) = 5.5
        let data = vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = compute_stats(&data);
        assert!((stats.q1     - 4.0).abs() < 1e-9, "Q1: {}", stats.q1);
        assert!((stats.median - 4.5).abs() < 1e-9, "median: {}", stats.median);
        assert!((stats.q3     - 5.5).abs() < 1e-9, "Q3: {}", stats.q3);
    }

    #[test]
    fn outliers_identified_correctly() {
        // [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        // Q1=2.25  Q3=7.75  IQR=5.5  hi_fence=7.75+8.25=16.0
        // 100 is an outlier; everything else is within fences.
        let data: Vec<f64> = vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,100.0];
        let stats = compute_stats(&data);
        assert_eq!(stats.outliers.len(), 1, "expected 1 outlier, got {:?}", stats.outliers);
        assert!((stats.outliers[0] - 100.0).abs() < 1e-9, "outlier should be 100");
    }

    #[test]
    fn no_outliers_when_all_data_within_fence() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let stats = compute_stats(&data);
        assert!(stats.outliers.is_empty(), "expected no outliers");
    }

    #[test]
    fn zero_iqr_group_has_identical_q1_median_q3() {
        // All-identical values: IQR = 0, fences = median ± 0.
        let data = vec![5.0_f64; 10];
        let stats = compute_stats(&data);
        assert!((stats.q1 - 5.0).abs() < 1e-9);
        assert!((stats.median - 5.0).abs() < 1e-9);
        assert!((stats.q3 - 5.0).abs() < 1e-9);
        assert!((stats.whisker_lo - 5.0).abs() < 1e-9);
        assert!((stats.whisker_hi - 5.0).abs() < 1e-9);
        assert!(stats.outliers.is_empty());
    }

    // ── build() happy path ────────────────────────────────────────────────────

    #[test]
    fn build_produces_svg_with_correct_structure() {
        let cats  = ["A","A","A","B","B","B","C","C","C"];
        let vals  = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0];
        let df = make_df(&cats, &vals);
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .build()
            .expect("build should succeed");
        let svg = chart.svg();
        assert!(svg.contains("<svg"), "output must be SVG");
        assert!(svg.contains("<rect"),  "must contain rect elements for boxes");
        assert!(svg.contains("<line"),  "must contain line elements for whiskers/median");
    }

    #[test]
    fn build_single_value_group_does_not_panic() {
        let df = make_df(&["A"], &[7.0]);
        BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .build()
            .expect("single-value group must build without panic");
    }

    #[test]
    fn build_zero_iqr_group_renders_flat_box() {
        // All five values identical → IQR = 0 → box has near-zero height.
        let df = make_df(&["A","A","A","A","A"], &[5.0,5.0,5.0,5.0,5.0]);
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .build()
            .expect("zero-IQR group must build");
        // Median line and whiskers still rendered; no division-by-zero crash.
        assert!(chart.svg().contains("<line"));
    }

    #[test]
    fn build_excludes_null_y_and_emits_warning() {
        let df = make_df_opt(
            &["A","A","A","A"],
            &[Some(1.0), None, Some(3.0), Some(5.0)],
        );
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .build()
            .expect("null y should be skipped, not error");
        let has_warning = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, .. } if col == "value"
        ));
        assert!(has_warning, "expected NullsSkipped warning");
    }

    // ── notched ───────────────────────────────────────────────────────────────

    #[test]
    fn notch_bounds_clamped_when_exceed_q1_q3_emits_warning() {
        // With N=2 identical-ish values the notch half-width is very large.
        let df = make_df(&["A","A"], &[4.9, 5.1]);
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .notched(true)
            .build()
            .expect("notched build must succeed");
        // With such a narrow IQR and small N the notch will certainly exceed Q1/Q3.
        let has_clamp_warn = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NotchClamped { group } if group == "A"
        ));
        assert!(has_clamp_warn, "expected NotchClamped warning for group A");
    }

    #[test]
    fn notch_not_clamped_for_large_n() {
        // Larger N → smaller notch half-width → no clamping needed.
        let cats: Vec<&str> = vec!["A"; 50];
        let vals: Vec<f64>  = (0..50).map(|i| i as f64).collect();
        let df = make_df(&cats, &vals);
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .notched(true)
            .build()
            .expect("large-N notched build must succeed");
        let has_clamp_warn = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NotchClamped { .. }
        ));
        assert!(!has_clamp_warn, "no clamping expected for large N");
    }

    // ── PointDisplay ──────────────────────────────────────────────────────────

    #[test]
    fn point_display_all_renders_more_circles_than_outliers() {
        // Values: 10 normal + 1 clear outlier at 1000.
        let cats: Vec<&str> = vec!["A"; 11];
        let mut vals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        vals.push(1000.0); // outlier
        let df = make_df(&cats, &vals);

        let chart_outliers = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .points(PointDisplay::Outliers)
            .build()
            .unwrap();

        let chart_all = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .points(PointDisplay::All)
            .build()
            .unwrap();

        let count_circles = |svg: &str| svg.matches("<circle").count();
        assert!(
            count_circles(chart_all.svg()) > count_circles(chart_outliers.svg()),
            "All mode must render more circles than Outliers mode"
        );
    }

    #[test]
    fn point_display_none_renders_no_circles() {
        let cats = ["A","A","A","A","A","A"];
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]; // one outlier
        let df = make_df(&cats, &vals);
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .points(PointDisplay::None)
            .build()
            .unwrap();
        assert_eq!(chart.svg().matches("<circle").count(), 0,
            "PointDisplay::None must render zero circles");
    }

    #[test]
    fn point_display_outliers_renders_only_outliers() {
        // One outlier at 1000; 5 normal values.
        let cats = ["A","A","A","A","A","A"];
        let vals = [1.0,2.0,3.0,4.0,5.0,1000.0];
        let df = make_df(&cats, &vals);
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .points(PointDisplay::Outliers)
            .build()
            .unwrap();
        assert_eq!(chart.svg().matches("<circle").count(), 1,
            "PointDisplay::Outliers must render exactly 1 circle for 1 outlier");
    }

    // ── Dimension and size ────────────────────────────────────────────────────

    #[test]
    fn build_produces_correct_dimensions() {
        let df = make_df(&["A","A","B","B"], &[1.0,2.0,3.0,4.0]);
        let chart = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .build()
            .unwrap();
        assert_eq!(chart.width(),  CANVAS_WIDTH);
        assert_eq!(chart.height(), CANVAS_HEIGHT);
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn build_numeric_x_returns_unsupported_column() {
        let df = DataFrame::new(vec![
            Series::new("x", &[1.0f64, 2.0]),
            Series::new("y", &[3.0f64, 4.0]),
        ]).unwrap();
        let err = BoxPlotBuilder::new(&df).x("x").y("y").build().unwrap_err();
        match err {
            CharcoalError::UnsupportedColumn { col, .. } => assert_eq!(col, "x"),
            other => panic!("expected UnsupportedColumn, got {other:?}"),
        }
    }

    #[test]
    fn build_categorical_y_returns_unsupported_column() {
        let df = DataFrame::new(vec![
            Series::new("cat",  &["A","B"]),
            Series::new("label",&["X","Y"]),
        ]).unwrap();
        let err = BoxPlotBuilder::new(&df).x("cat").y("label").build().unwrap_err();
        match err {
            CharcoalError::UnsupportedColumn { col, .. } => assert_eq!(col, "label"),
            other => panic!("expected UnsupportedColumn, got {other:?}"),
        }
    }

    #[test]
    fn build_exceeding_row_limit_returns_data_too_large() {
        let cats: Vec<&str> = vec!["A"; 10];
        let vals: Vec<f64>  = vec![1.0; 10];
        let df = make_df(&cats, &vals);
        let err = BoxPlotBuilder::new(&df)
            .x("group").y("value")
            .row_limit(5)
            .build()
            .unwrap_err();
        match err {
            CharcoalError::DataTooLarge { rows, limit, .. } => {
                assert_eq!(rows, 10);
                assert_eq!(limit, 5);
            }
            other => panic!("expected DataTooLarge, got {other:?}"),
        }
    }

    // ── Optional setters ─────────────────────────────────────────────────────

    #[test]
    fn title_setter_stores_string() {
        let df = DataFrame::empty();
        let b = BoxPlotBuilder::new(&df).title("My Chart");
        assert_eq!(b.config.title.as_deref(), Some("My Chart"));
    }

    #[test]
    fn notched_setter_stores_true() {
        let df = DataFrame::empty();
        let b = BoxPlotBuilder::new(&df).notched(true);
        assert!(b.config.notched);
    }

    #[test]
    fn points_setter_stores_variant() {
        let df = DataFrame::empty();
        let b = BoxPlotBuilder::new(&df).points(PointDisplay::All);
        assert!(matches!(b.config.points, PointDisplay::All));
    }

    #[test]
    fn row_limit_setter_stores_value() {
        let df = DataFrame::empty();
        let b = BoxPlotBuilder::new(&df).row_limit(500);
        assert_eq!(b.config.row_limit, 500);
    }

    // ── Jitter determinism ────────────────────────────────────────────────────

    #[test]
    fn jitter_is_deterministic_same_inputs() {
        let j1 = deterministic_jitter(2, 5, 20.0);
        let j2 = deterministic_jitter(2, 5, 20.0);
        assert_eq!(j1, j2, "jitter must be deterministic");
    }

    #[test]
    fn jitter_differs_between_values_in_same_group() {
        let j0 = deterministic_jitter(0, 0, 20.0);
        let j1 = deterministic_jitter(0, 1, 20.0);
        assert_ne!(j0, j1, "different value indices should yield different jitter");
    }

    #[test]
    fn jitter_magnitude_bounded_by_box_half() {
        let box_half = 15.0;
        for vi in 0..50 {
            let j = deterministic_jitter(0, vi, box_half);
            assert!(j.abs() <= box_half * 0.8 + 1e-9,
                "jitter {j} out of bounds for box_half={box_half}");
        }
    }
}