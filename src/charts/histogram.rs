//! Histogram chart builder (`Chart::histogram(&df)`).
//!
//! Follows the same typestate pattern as `scatter.rs`, `line.rs`, and `bar.rs`.
//!
//! # Column roles
//!
//! | Column | Required dtype | Role |
//! |--------|---------------|------|
//! | `x`    | Numeric       | Data values to be distributed into bins |
//!
//! A single required column produces its own y-axis from bin counts, so the
//! builder chain is shorter than Scatter/Line/Bar:
//! `HistogramBuilder` → (`.x()`) → `HistogramWithX` → (`.build()`)
//!
//! # Bin methods
//!
//! | Variant | Formula | Notes |
//! |---------|---------|-------|
//! | `Scott` (default) | `h = 3.49 σ n^{-1/3}` | Best for non-normal distributions |
//! | `Sturges`         | `k = ⌈log₂(n)⌉ + 1`   | Simple; under-bins for large n |
//! | `FreedmanDiaconis`| `h = 2 IQR n^{-1/3}`  | Robust to outliers |
//! | `Fixed(k)`        | `k` bins exactly       | Always takes precedence; set by `.bins(k)` |
//!
//! # Null handling
//!
//! Null values in the x column are excluded from all bin computation.
//! When at least one null exists the chart subtitle reads `"N null(s) excluded"`.
//! A [`CharcoalWarning::NullsSkipped`] is also emitted.
//!
//! # Normalization
//!
//! When `.normalize(true)` each bar height becomes
//! `count / (total_non_null × bin_width)`, a proper probability density.
//! The y-axis label switches from `"Count"` to `"Density"` automatically.
//!
//! # Rendering
//!
//! Bins are rendered as adjacent `<rect>` elements (no gap between them).
//! The x-axis is numeric, not categorical.

use polars::frame::DataFrame;

use crate::charts::Chart;
use crate::dtype::{classify_column, VizDtype};
use crate::error::{CharcoalError, CharcoalWarning};
use crate::normalize::to_f64;
use crate::render::{
    axes::{
        AxisOrientation, LinearScale,
        build_tick_marks, compute_axis,
        nice_ticks, tick_labels_numeric,
    },
    geometry,
    Margin, SvgCanvas,
};
use crate::theme::{Theme, ThemeConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CANVAS_WIDTH:      u32   = 800;
const CANVAS_HEIGHT:     u32   = 500;
const DEFAULT_ROW_LIMIT: usize = 1_000_000;
/// Y-axis headroom above the tallest bin (5 %).
const Y_HEADROOM: f64 = 1.05;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Strategy used to choose the number of histogram bins.
///
/// [`BinMethod::Fixed`] always takes precedence over all heuristic methods and
/// is what `.bins(n)` sets internally.
#[derive(Debug, Clone, PartialEq)]
pub enum BinMethod {
    /// Scott's rule: `h = 3.49 σ n^{-1/3}`.
    ///
    /// Default because it performs well on non-normal distributions common in
    /// data science.
    Scott,

    /// Sturges' rule: `k = ⌈log₂(n)⌉ + 1`.
    ///
    /// Simple but tends to under-bin for large or non-normal data.
    Sturges,

    /// Freedman-Diaconis rule: `h = 2 IQR n^{-1/3}`.
    ///
    /// Robust to outliers because it uses the interquartile range rather than
    /// the standard deviation.
    FreedmanDiaconis,

    /// Exact bin count supplied by the caller.
    ///
    /// Always used when set; ignores all statistical heuristics.
    /// This is what `.bins(n)` sets internally.
    Fixed(usize),
}

impl Default for BinMethod {
    fn default() -> Self {
        BinMethod::Scott
    }
}

// ---------------------------------------------------------------------------
// Accumulated configuration
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct HistogramConfig {
    pub x_col:      Option<String>,
    pub bin_method: BinMethod,
    pub normalize:  bool,
    pub title:      Option<String>,
    pub x_label:    Option<String>,
    pub y_label:    Option<String>,
    pub theme:      Theme,
    pub row_limit:  usize,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            x_col:      None,
            bin_method: BinMethod::default(),
            normalize:  false,
            title:      None,
            x_label:    None,
            y_label:    None,
            theme:      Theme::Default,
            row_limit:  DEFAULT_ROW_LIMIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder states
// ---------------------------------------------------------------------------

/// Initial histogram builder — `x` column not yet set.
pub struct HistogramBuilder<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: HistogramConfig,
}

/// `x` column set — `.build()` is now available.
pub struct HistogramWithX<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: HistogramConfig,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'df> HistogramBuilder<'df> {
    pub(crate) fn new(df: &'df DataFrame) -> Self {
        Self { df, config: HistogramConfig::default() }
    }
}

// ---------------------------------------------------------------------------
// Optional setter methods — stamped onto both states via macro so the
// compiler sees each as returning `Self`, preserving the typestate.
// ---------------------------------------------------------------------------

macro_rules! impl_histogram_optional_setters {
    ($t:ty) => {
        impl<'df> $t {
            /// Chart title rendered above the plot area.
            pub fn title(mut self, title: &str) -> Self {
                self.config.title = Some(title.to_string());
                self
            }

            /// Label rendered below the x-axis. Defaults to the column name.
            pub fn x_label(mut self, label: &str) -> Self {
                self.config.x_label = Some(label.to_string());
                self
            }

            /// Label rendered beside the y-axis.
            ///
            /// Defaults to `"Count"` (or `"Density"` when `.normalize(true)` is set).
            /// Explicit values passed here override the automatic label.
            pub fn y_label(mut self, label: &str) -> Self {
                self.config.y_label = Some(label.to_string());
                self
            }

            /// Visual theme. Defaults to [`Theme::Default`].
            pub fn theme(mut self, theme: Theme) -> Self {
                self.config.theme = theme;
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

impl_histogram_optional_setters!(HistogramBuilder<'df>);
impl_histogram_optional_setters!(HistogramWithX<'df>);

// ---------------------------------------------------------------------------
// Required field transition
// ---------------------------------------------------------------------------

impl<'df> HistogramBuilder<'df> {
    /// Set the x-axis column. Column name is validated at `.build()`, not here.
    pub fn x(mut self, col: &str) -> HistogramWithX<'df> {
        self.config.x_col = Some(col.to_string());
        HistogramWithX { df: self.df, config: self.config }
    }
}

// ---------------------------------------------------------------------------
// Histogram-specific setters and .build() — available only on HistogramWithX
// ---------------------------------------------------------------------------

impl<'df> HistogramWithX<'df> {
    /// Select the binning strategy. Defaults to [`BinMethod::Scott`].
    ///
    /// See [`BinMethod`] for the available options. Calling `.bins(n)` is
    /// shorthand for `.bin_method(BinMethod::Fixed(n))`.
    pub fn bin_method(mut self, method: BinMethod) -> Self {
        self.config.bin_method = method;
        self
    }

    /// Set an exact bin count, overriding all heuristic methods.
    ///
    /// Equivalent to `.bin_method(BinMethod::Fixed(n))`.
    pub fn bins(mut self, n: usize) -> Self {
        self.config.bin_method = BinMethod::Fixed(n);
        self
    }

    /// When `true`, the y-axis shows probability density rather than raw counts.
    ///
    /// Density = `count / (total_non_null × bin_width)`.
    /// The y-axis label switches from `"Count"` to `"Density"` automatically
    /// (unless overridden by `.y_label()`).
    pub fn normalize(mut self, norm: bool) -> Self {
        self.config.normalize = norm;
        self
    }

    /// Validate inputs, compute bins, render bin rectangles, and return a [`Chart`].
    ///
    /// # Errors
    ///
    /// - [`CharcoalError::DataTooLarge`] if `df.height() > row_limit`.
    /// - [`CharcoalError::ColumnNotFound`] if the x column name does not exist.
    /// - [`CharcoalError::UnsupportedColumn`] if the x column is not Numeric.
    /// - [`CharcoalError::InsufficientData`] if all x values are null.
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
        // 2. Validate x column (must be Numeric)
        // ------------------------------------------------------------------
        let x_col = config.x_col.as_deref().unwrap(); // always present (typestate)

        let x_viz = classify_column(df, x_col, None)?;
        if x_viz != VizDtype::Numeric {
            let dtype = df.schema().get(x_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     x_col.to_string(),
                dtype,
                message: format!(
                    "Histogram x column must be Numeric, but \"{x_col}\" \
                     is {x_viz:?}. Cast the column to a numeric type before charting.",
                ),
            });
        }

        // ------------------------------------------------------------------
        // 3. Normalize to f64, preserving nulls
        // ------------------------------------------------------------------
        let (raw_vals, x_warnings) = to_f64(df, x_col)?;
        warnings.extend(x_warnings);

        let null_count = raw_vals.iter().filter(|v| v.is_none()).count();
        let values: Vec<f64> = raw_vals.into_iter().flatten().collect();

        if values.is_empty() {
            return Err(CharcoalError::InsufficientData {
                col:      x_col.to_string(),
                required: 1,
                got:      0,
            });
        }

        // ------------------------------------------------------------------
        // 4. Compute bins
        // ------------------------------------------------------------------
        let (bin_count, bin_width, bin_edges) = compute_bins(&values, &config.bin_method)?;

        // ------------------------------------------------------------------
        // 5. Assign values to bins → raw counts
        // ------------------------------------------------------------------
        let counts = assign_to_bins(&values, bin_count, &bin_edges);

        // ------------------------------------------------------------------
        // 6. Compute displayed y values (raw count or probability density)
        // ------------------------------------------------------------------
        let total_non_null = values.len() as f64;
        let y_values: Vec<f64> = if config.normalize {
            counts
                .iter()
                .map(|&c| c as f64 / (total_non_null * bin_width))
                .collect()
        } else {
            counts.iter().map(|&c| c as f64).collect()
        };

        // ------------------------------------------------------------------
        // 7. Axis labels and subtitle
        // ------------------------------------------------------------------
        let x_label = config.x_label.as_deref().unwrap_or(x_col);

        let auto_y_label = if config.normalize { "Density" } else { "Count" };
        let y_label = config.y_label.as_deref().unwrap_or(auto_y_label);

        // Subtitle notes excluded nulls.
        let subtitle: String = if null_count > 0 {
            format!("{null_count} null{} excluded", if null_count == 1 { "" } else { "s" })
        } else {
            String::new()
        };

        // The canvas title: user title + optional subtitle on a separate line.
        let display_title = match (config.title.as_deref(), subtitle.is_empty()) {
            (Some(t), true)  => t.to_string(),
            (Some(t), false) => format!("{t}\n{subtitle}"),
            (None,    true)  => String::new(),
            (None,    false) => subtitle.clone(),
        };

        // ------------------------------------------------------------------
        // 8. Axis ranges → nice ticks
        // ------------------------------------------------------------------
        let x_min = bin_edges[0];
        let x_max = bin_edges[bin_count];

        let y_max_data = y_values.iter().cloned().fold(0.0_f64, f64::max);
        let y_max_data = if y_max_data == 0.0 { 1.0 } else { y_max_data };
        let y_axis_max = y_max_data * Y_HEADROOM;

        let x_tick_vals = nice_ticks(x_min, x_max, 6);
        let y_tick_vals = nice_ticks(0.0, y_axis_max, 6);

        // ------------------------------------------------------------------
        // 9. Canvas + scales
        // ------------------------------------------------------------------
        let theme_cfg = ThemeConfig::from(&config.theme);
        // Capture the bar fill color before theme_cfg is moved into the canvas.
        let bar_color = theme_cfg.palette[0];
        let canvas = SvgCanvas::new(
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
            Margin::default_chart(),
            theme_cfg,
        );
        let ox = canvas.plot_origin_x();
        let oy = canvas.plot_origin_y();
        let pw = canvas.plot_width();
        let ph = canvas.plot_height();

        // x: left → right.  y: data 0 → pixel bottom, data max → pixel top.
        let x_scale = LinearScale::new(
            *x_tick_vals.first().unwrap(),
            *x_tick_vals.last().unwrap(),
            ox,
            ox + pw,
        );
        let y_scale = LinearScale::new(
            *y_tick_vals.first().unwrap(),
            *y_tick_vals.last().unwrap(),
            oy + ph, // pixel bottom  =  data 0
            oy,      // pixel top     =  data max
        );

        // ------------------------------------------------------------------
        // 10. Render bin rectangles
        //
        // Each bin spans from bin_edges[i] to bin_edges[i+1].
        // Adjacent rects share an x-boundary exactly — there is no gap.
        // ------------------------------------------------------------------
        let baseline_px = y_scale.map(0.0);
        let mut elements: Vec<String> = Vec::with_capacity(bin_count);

        for i in 0..bin_count {
            let left_px  = x_scale.map(bin_edges[i]);
            let right_px = x_scale.map(bin_edges[i + 1]);
            let rect_w   = (right_px - left_px).max(1.0); // ≥ 1 px: always visible

            let top_px = y_scale.map(y_values[i]);
            let rect_h = (baseline_px - top_px).abs();

            if rect_h > 0.0 {
                elements.push(geometry::rect(
                    left_px,
                    top_px,
                    rect_w,
                    rect_h,
                    bar_color,
                    0.0, // rx = 0 — no rounded corners
                ));
            }
        }

        // ------------------------------------------------------------------
        // 11. Build axis SVG output
        // ------------------------------------------------------------------
        let x_labels = tick_labels_numeric(&x_tick_vals);
        let y_labels = tick_labels_numeric(&y_tick_vals);
        let x_ticks  = build_tick_marks(&x_tick_vals, &x_labels, &x_scale);
        let y_ticks  = build_tick_marks(&y_tick_vals, &y_labels, &y_scale);

        let x_axis = compute_axis(
            &x_scale, &x_ticks, AxisOrientation::Horizontal,
            ox, oy, pw, ph, &canvas.theme,
        );
        let y_axis = compute_axis(
            &y_scale, &y_ticks, AxisOrientation::Vertical,
            ox, oy, pw, ph, &canvas.theme,
        );

        // ------------------------------------------------------------------
        // 12. Assemble Chart — no legend for histograms
        // ------------------------------------------------------------------
        let svg = canvas.render(
            elements,
            x_axis,
            y_axis,
            &display_title,
            x_label,
            y_label,
            None,
        );

        Ok(Chart {
            svg,
            warnings,
            title:  config.title.as_deref().unwrap_or("").to_string(),
            width:  CANVAS_WIDTH,
            height: CANVAS_HEIGHT,
        })
    }
}

// ---------------------------------------------------------------------------
// Bin computation
// ---------------------------------------------------------------------------

/// Compute `(bin_count, bin_width, bin_edges)` from non-null values and method.
///
/// `bin_edges` has length `bin_count + 1`:
/// - `edges[i]`          — left  boundary of bin `i`
/// - `edges[bin_count]`  — right boundary of the last bin
///
/// # Degenerate input
///
/// When all values are identical (`data_range == 0`), the function returns a
/// single bin of width `1.0` centred on the value so that `bin_width > 0`
/// and no division by zero occurs anywhere downstream.
pub(crate) fn compute_bins(
    values: &[f64],
    method: &BinMethod,
) -> Result<(usize, f64, Vec<f64>), CharcoalError> {
    debug_assert!(!values.is_empty(), "compute_bins requires at least one value");

    let n     = values.len();
    let x_min = values.iter().cloned().fold(f64::INFINITY,     f64::min);
    let x_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = x_max - x_min;

    let bin_count: usize = match method {
        BinMethod::Fixed(k) => {
            if *k == 0 {
                return Err(CharcoalError::InsufficientData {
                    col:      "bins".to_string(),
                    required: 1,
                    got:      0,
                });
            }
            *k
        }

        BinMethod::Sturges => {
            // k = ⌈log₂(n)⌉ + 1
            ((n as f64).log2().ceil() as usize + 1).max(1)
        }

        BinMethod::Scott => {
            if range == 0.0 {
                1
            } else {
                let sigma = sample_std_dev(values);
                if sigma == 0.0 {
                    1
                } else {
                    // h = 3.49 σ n^{-1/3}  →  k = ⌈range / h⌉
                    let h = 3.49 * sigma * (n as f64).powf(-1.0 / 3.0);
                    ((range / h).ceil() as usize).max(1)
                }
            }
        }

        BinMethod::FreedmanDiaconis => {
            if range == 0.0 {
                1
            } else {
                let iqr = interquartile_range(values);
                if iqr == 0.0 {
                    // IQR is zero (constant-ish distribution) — fall back to Sturges.
                    ((n as f64).log2().ceil() as usize + 1).max(1)
                } else {
                    // h = 2 IQR n^{-1/3}  →  k = ⌈range / h⌉
                    let h = 2.0 * iqr * (n as f64).powf(-1.0 / 3.0);
                    ((range / h).ceil() as usize).max(1)
                }
            }
        }
    };

    // When all values are identical, widen the axis so the canvas is non-degenerate.
    let (eff_min, eff_max) = if range == 0.0 {
        (x_min - 0.5, x_min + 0.5)
    } else {
        (x_min, x_max)
    };

    let eff_range = eff_max - eff_min;
    let bin_width = eff_range / bin_count as f64;

    let bin_edges: Vec<f64> = (0..=bin_count)
        .map(|i| eff_min + i as f64 * bin_width)
        .collect();

    Ok((bin_count, bin_width, bin_edges))
}

/// Assign each value to a bin; return a `Vec<usize>` of length `bin_count`.
///
/// A value exactly equal to `bin_edges[bin_count]` (the right edge of the last
/// bin) is placed in the final bin rather than overflowing.
pub(crate) fn assign_to_bins(
    values:    &[f64],
    bin_count: usize,
    bin_edges: &[f64],
) -> Vec<usize> {
    let mut counts = vec![0usize; bin_count];

    let x_min = bin_edges[0];
    let x_max = bin_edges[bin_count];
    let total = x_max - x_min;

    for &v in values {
        let idx = if total == 0.0 {
            0
        } else {
            let raw = ((v - x_min) / total * bin_count as f64) as usize;
            raw.min(bin_count - 1) // clamp right-boundary value into last bin
        };
        counts[idx] += 1;
    }

    counts
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Sample standard deviation using Bessel's correction (n − 1 denominator).
///
/// Returns `0.0` for slices with fewer than 2 elements.
fn sample_std_dev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 { return 0.0; }
    let mean     = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt()
}

/// Interquartile range (Q3 − Q1) using linear interpolation.
///
/// Sorts a copy of the input; the caller's slice is not mutated.
fn interquartile_range(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, 0.75) - percentile_sorted(&sorted, 0.25)
}

/// Linear-interpolation percentile on an already-sorted slice.
///
/// Uses `floor((N − 1) · p)` as the lower index, consistent with the
/// box-plot percentile convention in Step 2.7.
pub(crate) fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n == 1 { return sorted[0]; }
    let pos  = p * (n - 1) as f64;
    let lo   = pos.floor() as usize;
    let hi   = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    // ------------------------------------------------------------------
    // Shared fixtures
    // ------------------------------------------------------------------

    fn x_df(values: &[f64]) -> DataFrame {
        DataFrame::new(vec![Series::new("x", values)]).unwrap()
    }

    fn x_df_with_nulls() -> DataFrame {
        let s = Series::new("x", &[Some(1.0f64), None, Some(3.0), None, Some(5.0)]);
        DataFrame::new(vec![s]).unwrap()
    }

    /// Deterministic pseudo-normal sample via Box-Muller + LCG (no rand dep).
    fn pseudo_normal(n: usize, seed: u64) -> Vec<f64> {
        let lcg = |s: u64| s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut state = seed;
        let mut out   = Vec::with_capacity(n);
        while out.len() < n {
            state = lcg(state);
            let u1 = ((state >> 11) as f64 / (1u64 << 53) as f64).max(f64::EPSILON);
            state = lcg(state);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
            out.push(z0);
            if out.len() < n { out.push(z1); }
        }
        out
    }

    // ------------------------------------------------------------------
    // 2.5.1  BinMethod default is Scott
    // ------------------------------------------------------------------

    #[test]
    fn bin_method_default_is_scott() {
        assert_eq!(BinMethod::default(), BinMethod::Scott);
    }

    // ------------------------------------------------------------------
    // 2.5.2  Scott's rule produces a reasonable bin count for a normal dist
    // ------------------------------------------------------------------

    #[test]
    fn scotts_rule_reasonable_bin_count_for_normal_n1000() {
        // For n=1000, σ≈1: h = 3.49/10 ≈ 0.349.  range ≈ 6–8 → ~17–23 bins.
        // We allow 10–50 to stay robust across different seeds.
        let values = pseudo_normal(1000, 42);
        let (bin_count, _bw, edges) = compute_bins(&values, &BinMethod::Scott).unwrap();
        assert!(
            (10..=50).contains(&bin_count),
            "Scott's rule gave {bin_count} bins for n=1000 normal — expected 10–50"
        );
        assert_eq!(edges.len(), bin_count + 1);
    }

    // ------------------------------------------------------------------
    // 2.5.2  Fixed(20) produces exactly 20 bins
    // ------------------------------------------------------------------

    #[test]
    fn fixed_20_produces_exactly_20_bins() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (bin_count, _bw, edges) = compute_bins(&values, &BinMethod::Fixed(20)).unwrap();
        assert_eq!(bin_count, 20);
        assert_eq!(edges.len(), 21);
    }

    #[test]
    fn bins_setter_is_shorthand_for_fixed() {
        let df = x_df(&(0..100).map(|i| i as f64).collect::<Vec<_>>());
        let c1 = HistogramBuilder::new(&df).x("x").bins(15).build().unwrap();
        let c2 = HistogramBuilder::new(&df).x("x").bin_method(BinMethod::Fixed(15)).build().unwrap();
        // Same number of rendered data rects.
        assert_eq!(
            count_data_rects(c1.svg()),
            count_data_rects(c2.svg()),
            ".bins(15) and .bin_method(Fixed(15)) must produce the same output"
        );
    }

    // ------------------------------------------------------------------
    // Fixed(0) is an error
    // ------------------------------------------------------------------

    #[test]
    fn fixed_zero_returns_error() {
        let values = vec![1.0, 2.0, 3.0];
        assert!(compute_bins(&values, &BinMethod::Fixed(0)).is_err());
    }

    // ------------------------------------------------------------------
    // Bin counts sum to the number of non-null values
    // ------------------------------------------------------------------

    #[test]
    fn bin_counts_sum_to_total_non_null() {
        let values: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
        let (bin_count, _bw, edges) = compute_bins(&values, &BinMethod::Fixed(10)).unwrap();
        let counts = assign_to_bins(&values, bin_count, &edges);
        assert_eq!(counts.iter().sum::<usize>(), values.len());
    }

    // ------------------------------------------------------------------
    // 2.5.2  Null values are excluded and noted in the SVG subtitle
    // ------------------------------------------------------------------

    #[test]
    fn null_values_produce_subtitle_in_svg() {
        let df    = x_df_with_nulls(); // 2 nulls
        let chart = HistogramBuilder::new(&df).x("x").bins(3).build().unwrap();
        assert!(
            chart.svg().contains("2 nulls excluded"),
            "SVG must note the 2 excluded nulls"
        );
    }

    #[test]
    fn single_null_uses_singular_form() {
        let s  = Series::new("x", &[Some(1.0f64), None, Some(3.0)]);
        let df = DataFrame::new(vec![s]).unwrap();
        let chart = HistogramBuilder::new(&df).x("x").bins(2).build().unwrap();
        assert!(
            chart.svg().contains("1 null excluded"),
            "Single null must use singular 'null excluded'"
        );
        assert!(!chart.svg().contains("1 nulls excluded"),
            "Must not use plural 'nulls' for a single null");
    }

    #[test]
    fn no_subtitle_when_no_nulls() {
        let df    = x_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let chart = HistogramBuilder::new(&df).x("x").bins(3).build().unwrap();
        assert!(!chart.svg().contains("excluded"));
    }

    // ------------------------------------------------------------------
    // 2.5.3  Normalized output: ∑(density × bin_width) ≈ 1
    // ------------------------------------------------------------------

    #[test]
    fn normalized_density_integrates_to_one() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let (bin_count, bin_width, bin_edges) =
            compute_bins(&vals, &BinMethod::Fixed(10)).unwrap();
        let counts  = assign_to_bins(&vals, bin_count, &bin_edges);
        let n       = vals.len() as f64;
        let integral: f64 = counts
            .iter()
            .map(|&c| (c as f64 / (n * bin_width)) * bin_width)
            .sum();
        assert!(
            (integral - 1.0).abs() < 1e-9,
            "Normalized histogram integral = {integral:.12}, expected ≈ 1.0"
        );
    }

    #[test]
    fn normalize_true_changes_y_axis_label_to_density() {
        let df    = x_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let chart = HistogramBuilder::new(&df).x("x").normalize(true).bins(3).build().unwrap();
        assert!(chart.svg().contains("Density"),
            "Normalized histogram must show 'Density' on the y-axis");
    }

    #[test]
    fn normalize_false_uses_count_label() {
        let df    = x_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let chart = HistogramBuilder::new(&df).x("x").normalize(false).bins(3).build().unwrap();
        assert!(chart.svg().contains("Count"),
            "Non-normalized histogram must show 'Count' on the y-axis");
    }

    // ------------------------------------------------------------------
    // 2.5.4  Rendering — bins are adjacent `<rect>` elements; x-axis numeric
    // ------------------------------------------------------------------

    #[test]
    fn svg_is_well_formed() {
        let df  = x_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let svg = HistogramBuilder::new(&df).x("x").build().unwrap().svg().to_string();
        assert!(svg.starts_with("<svg"),  "must start with <svg");
        assert!(svg.ends_with("</svg>"), "must end with </svg>");
    }

    #[test]
    fn canvas_dimensions_are_800_x_500() {
        let df    = x_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let chart = HistogramBuilder::new(&df).x("x").build().unwrap();
        assert_eq!(chart.width(),  800);
        assert_eq!(chart.height(), 500);
    }

    #[test]
    fn fixed_10_bins_produces_at_most_10_data_rects() {
        let df    = x_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let chart = HistogramBuilder::new(&df).x("x").bins(10).build().unwrap();
        let n     = count_data_rects(chart.svg());
        assert!(n <= 10 && n >= 1,
            "Fixed(10) must produce 1–10 data rects; got {n}");
    }

    #[test]
    fn bins_are_adjacent_no_gap() {
        let vals: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let df    = x_df(&vals);
        let chart = HistogramBuilder::new(&df).x("x").bins(5).build().unwrap();
        let rects = extract_data_rects(chart.svg());
        assert!(rects.len() > 1, "Need > 1 rect to test adjacency");

        for i in 0..rects.len() - 1 {
            let right_i = rects[i].0 + rects[i].1; // x + width
            let left_j  = rects[i + 1].0;
            assert!(
                (right_i - left_j).abs() < 0.5,
                "Gap between bin {i} and bin {}: right={right_i:.3} left={left_j:.3}",
                i + 1
            );
        }
    }

    #[test]
    fn x_axis_has_numeric_tick_labels() {
        let vals: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let svg = HistogramBuilder::new(&x_df(&vals)).x("x").build().unwrap().svg().to_string();
        // At least one numeric tick like >0< or >5< or >10< must appear.
        let has_numeric = svg.contains(">0<") || svg.contains(">5<") || svg.contains(">10<");
        assert!(has_numeric, "x-axis must have numeric tick labels");
    }

    #[test]
    fn no_legend_rendered() {
        let df  = x_df(&[1.0, 2.0, 3.0]);
        let svg = HistogramBuilder::new(&df).x("x").build().unwrap().svg().to_string();
        // Legend swatches use rx="…" attribute — must be absent.
        assert!(!svg.contains("rx=\""), "histograms must not render a legend");
    }

    #[test]
    fn all_rects_share_the_same_baseline() {
        // Every bin bottom must equal y_scale.map(0) = oy + ph = 440.
        let vals: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let chart = HistogramBuilder::new(&x_df(&vals)).x("x").bins(5).build().unwrap();
        let rects_yh = extract_data_rects_yh(chart.svg());
        assert!(!rects_yh.is_empty());
        for (i, &(y, h)) in rects_yh.iter().enumerate() {
            let bottom = y + h;
            assert!(
                (bottom - 440.0).abs() < 0.5,
                "bin {i} bottom must equal 440 (oy+ph); got {bottom:.3}"
            );
        }
    }

    // ------------------------------------------------------------------
    // 2.5.5  Single-value column (zero variance) does not divide by zero
    // ------------------------------------------------------------------

    #[test]
    fn single_unique_value_does_not_panic_scott() {
        let values = vec![42.0_f64; 50];
        let (bin_count, bin_width, edges) = compute_bins(&values, &BinMethod::Scott).unwrap();
        assert_eq!(bin_count, 1);
        assert!(bin_width > 0.0);
        assert_eq!(edges.len(), 2);
        let counts = assign_to_bins(&values, bin_count, &edges);
        assert_eq!(counts[0], 50);
    }

    #[test]
    fn single_unique_value_builds_without_panic() {
        let df = x_df(&[7.0f64; 20]);
        HistogramBuilder::new(&df).x("x").build()
            .expect("zero-variance column must build successfully");
    }

    // ------------------------------------------------------------------
    // Sturges known value (n=64 → 7 bins)
    // ------------------------------------------------------------------

    #[test]
    fn sturges_known_bin_count_n64() {
        let values: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let (k, _, _) = compute_bins(&values, &BinMethod::Sturges).unwrap();
        assert_eq!(k, 7, "Sturges for n=64 must give 7 bins");
    }

    // ------------------------------------------------------------------
    // FreedmanDiaconis reasonable range
    // ------------------------------------------------------------------

    #[test]
    fn fd_reasonable_bins_for_uniform_n200() {
        let values: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let (k, _, _) = compute_bins(&values, &BinMethod::FreedmanDiaconis).unwrap();
        assert!(
            (5..=100).contains(&k),
            "FreedmanDiaconis for n=200 uniform gave {k} bins — expected 5–100"
        );
    }

    // ------------------------------------------------------------------
    // Statistical helpers
    // ------------------------------------------------------------------

    #[test]
    fn sample_std_dev_known_value() {
        let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = sample_std_dev(&v);
        assert!((sd - 2.138_89).abs() < 0.001, "sample_std_dev = {sd:.6}");
    }

    #[test]
    fn sample_std_dev_single_element_is_zero() {
        assert_eq!(sample_std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn percentile_sorted_median_five_elements() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_sorted(&sorted, 0.5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn iqr_known_value_1_to_8() {
        // Q1=2.75, Q3=6.25, IQR=3.5
        let v: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let iqr = interquartile_range(&v);
        assert!((iqr - 3.5).abs() < 1e-9, "IQR = {iqr:.9}, expected 3.5");
    }

    // ------------------------------------------------------------------
    // Column validation
    // ------------------------------------------------------------------

    #[test]
    fn missing_column_returns_column_not_found() {
        let df  = x_df(&[1.0, 2.0]);
        let err = HistogramBuilder::new(&df).x("y").build().unwrap_err();
        assert!(
            matches!(err, CharcoalError::ColumnNotFound { ref name, .. } if name == "y"),
            "expected ColumnNotFound for 'y'; got {err:?}"
        );
    }

    #[test]
    fn categorical_x_returns_unsupported_column() {
        let df = DataFrame::new(vec![Series::new("cat", &["a", "b", "c"])]).unwrap();
        let err = HistogramBuilder::new(&df).x("cat").build().unwrap_err();
        assert!(
            matches!(err, CharcoalError::UnsupportedColumn { ref col, .. } if col == "cat"),
            "expected UnsupportedColumn for 'cat'; got {err:?}"
        );
    }

    #[test]
    fn all_null_column_returns_insufficient_data() {
        let s   = Series::new("x", &[None::<f64>, None, None]);
        let df  = DataFrame::new(vec![s]).unwrap();
        let err = HistogramBuilder::new(&df).x("x").build().unwrap_err();
        assert!(
            matches!(err, CharcoalError::InsufficientData { .. }),
            "all-null column must return InsufficientData; got {err:?}"
        );
    }

    #[test]
    fn row_limit_exceeded_returns_data_too_large() {
        let df  = x_df(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let err = HistogramBuilder::new(&df).x("x").row_limit(3).build().unwrap_err();
        match err {
            CharcoalError::DataTooLarge { rows, limit, .. } => {
                assert_eq!(rows,  5);
                assert_eq!(limit, 3);
            }
            other => panic!("expected DataTooLarge; got {other:?}"),
        }
    }

    #[test]
    fn row_limit_exactly_equal_does_not_error() {
        let df = x_df(&[1.0, 2.0, 3.0]);
        HistogramBuilder::new(&df).x("x").row_limit(3).build()
            .expect("row_limit == n_rows must succeed");
    }

    // ------------------------------------------------------------------
    // Optional setters survive the typestate transition
    // ------------------------------------------------------------------

    #[test]
    fn optional_setters_survive_x_transition() {
        let df = x_df(&[1.0, 2.0, 3.0]);
        let b = HistogramBuilder::new(&df)
            .title("My Histogram")
            .x("x")
            .bins(5)
            .theme(Theme::Dark);
        assert_eq!(b.config.title.as_deref(), Some("My Histogram"));
        assert!(matches!(b.config.bin_method, BinMethod::Fixed(5)));
        assert!(matches!(b.config.theme, Theme::Dark));
    }

    #[test]
    fn title_appears_in_svg() {
        let df  = x_df(&[1.0, 2.0, 3.0]);
        let svg = HistogramBuilder::new(&df).x("x").title("My Chart").build().unwrap().svg().to_string();
        assert!(svg.contains("My Chart"), "title must appear in the SVG");
    }

    #[test]
    fn df_reference_is_not_cloned() {
        let df = x_df(&[1.0, 2.0, 3.0]);
        let b  = HistogramBuilder::new(&df)
            .title("T")
            .x("x")
            .bins(3)
            .theme(Theme::Minimal)
            .row_limit(100_000);
        assert!(std::ptr::eq(b.df, &df));
    }

    // ------------------------------------------------------------------
    // SVG-parsing utilities used by tests above
    // ------------------------------------------------------------------

    /// Count data `<rect` tokens (excludes background and legend swatches).
    fn count_data_rects(svg: &str) -> usize {
        extract_data_rects(svg).len()
    }

    /// Extract `(x, width)` from every data `<rect` token.
    ///
    /// Excludes the canvas background, the clipPath rect inside `<defs>`, and
    /// legend swatches (identified by an `rx` attribute).
    fn extract_data_rects(svg: &str) -> Vec<(f64, f64)> {
        let stripped: String = if let (Some(ds), Some(de)) =
            (svg.find("<defs>"), svg.find("</defs>"))
        {
            format!("{}{}", &svg[..ds], &svg[de + 7..])
        } else {
            svg.to_string()
        };
        let mut out  = Vec::new();
        let mut rest = stripped.as_str();
        while let Some(start) = rest.find("<rect ") {
            rest = &rest[start..];
            let end = rest.find("/>").expect("unclosed <rect");
            let tok = &rest[..end + 2];
            let w = parse_attr(tok, "width").unwrap_or(0.0);
            let h = parse_attr(tok, "height").unwrap_or(0.0);
            // Skip canvas background (800×500) and legend swatches (rx attribute)
            if w < 800.0 && h < 500.0 && !tok.contains("rx=\"") {
                let x = parse_attr(tok, "x").unwrap_or(0.0);
                out.push((x, w));
            }
            rest = &rest[end + 2..];
        }
        out
    }

    /// Extract `(y, height)` from every data `<rect` token.
    ///
    /// Applies the same exclusion rules as `extract_data_rects`.
    fn extract_data_rects_yh(svg: &str) -> Vec<(f64, f64)> {
        let stripped: String = if let (Some(ds), Some(de)) =
            (svg.find("<defs>"), svg.find("</defs>"))
        {
            format!("{}{}", &svg[..ds], &svg[de + 7..])
        } else {
            svg.to_string()
        };
        let mut out  = Vec::new();
        let mut rest = stripped.as_str();
        while let Some(start) = rest.find("<rect ") {
            rest = &rest[start..];
            let end = rest.find("/>").expect("unclosed <rect");
            let tok = &rest[..end + 2];
            let w = parse_attr(tok, "width").unwrap_or(0.0);
            let h = parse_attr(tok, "height").unwrap_or(0.0);
            if w < 800.0 && h < 500.0 && !tok.contains("rx=\"") {
                let y = parse_attr(tok, "y").unwrap_or(0.0);
                out.push((y, h));
            }
            rest = &rest[end + 2..];
        }
        out
    }

    fn parse_attr(s: &str, name: &str) -> Option<f64> {
        let needle = format!("{name}=\"");
        let start  = s.find(&needle)? + needle.len();
        let end    = s[start..].find('"')?;
        s[start..start + end].parse().ok()
    }
}