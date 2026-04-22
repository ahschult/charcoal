//! Line chart builder (`Chart::line(&df)`).
//!
//! Follows the same typestate pattern as `scatter.rs`. The key additions over Scatter are:
//!
//! - **[`NullPolicy`]** — `Skip` (default) ends the current polyline at a null y-value and
//!   starts a new one after the gap; `Interpolate` fills the gap using linear interpolation
//!   from the nearest non-null neighbours on each side.
//! - **[`DashStyle`]** — maps to the SVG `stroke-dasharray` attribute of every rendered path.
//! - **Multi-series** — when `color_by` names a Categorical column, one polyline series is
//!   rendered per unique category value, each coloured from the theme palette.
//!
//! # Null semantics
//!
//! | Column | Behaviour |
//! |--------|-----------|
//! | `x` (any policy) | Row dropped; `NullsSkipped` warning emitted for the x column. |
//! | `y` under `Skip` | Current path segment ends; a new one starts after the gap. |
//! | `y` under `Interpolate` | Gap filled by x-distance-weighted linear interpolation. Leading / trailing nulls (no neighbour on one side) remain as gaps rather than being extrapolated. |
//!
//! A [`CharcoalWarning::NullsSkipped`] warning is emitted for every null y value regardless
//! of `NullPolicy` — `Interpolate` fills the gap visually but does not suppress the warning.

use polars::frame::DataFrame;

use crate::charts::{Chart, DashStyle, NullPolicy};
use crate::dtype::{classify_column, VizDtype};
use crate::error::{CharcoalError, CharcoalWarning};
use crate::normalize::{to_epoch_ms, to_f64, to_string};
use crate::render::{
    SvgCanvas, Margin,
    axes::{
        AxisOrientation, LinearScale, build_tick_marks, compute_axis,
        nice_ticks, tick_labels_numeric, tick_labels_temporal,
    },
    geometry,
};
use crate::theme::{Theme, ThemeConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CANVAS_WIDTH:      u32   = 800;
const CANVAS_HEIGHT:     u32   = 500;
const DEFAULT_STROKE:    f64   = 2.0;
const DEFAULT_ROW_LIMIT: usize = 1_000_000;
/// Neutral grey for the synthetic null-category series. Matches scatter.rs.
const NULL_COLOR: &str = "#AAAAAA";

// ---------------------------------------------------------------------------
// Accumulated configuration
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct LineConfig {
    pub x_col:        Option<String>,
    pub y_col:        Option<String>,
    pub color_by:     Option<String>,
    pub title:        Option<String>,
    pub x_label:      Option<String>,
    pub y_label:      Option<String>,
    pub theme:        Theme,
    pub null_policy:  NullPolicy,
    pub dash_style:   DashStyle,
    pub stroke_width: f64,
    pub row_limit:    usize,
}

impl Default for LineConfig {
    fn default() -> Self {
        Self {
            x_col:        None,
            y_col:        None,
            color_by:     None,
            title:        None,
            x_label:      None,
            y_label:      None,
            theme:        Theme::Default,
            null_policy:  NullPolicy::Skip,
            dash_style:   DashStyle::Solid,
            stroke_width: DEFAULT_STROKE,
            row_limit:    DEFAULT_ROW_LIMIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder states
// ---------------------------------------------------------------------------

/// Initial line builder — no required fields set yet.
pub struct LineBuilder<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: LineConfig,
}

/// x-axis column set; y still required.
pub struct LineWithX<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: LineConfig,
}

/// Both required fields set — `.build()` is available.
pub struct LineWithXY<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: LineConfig,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'df> LineBuilder<'df> {
    pub(crate) fn new(df: &'df DataFrame) -> Self {
        Self { df, config: LineConfig::default() }
    }
}

// ---------------------------------------------------------------------------
// Optional setter methods — stamped onto all three states via macro so
// the compiler sees each as returning `Self`, preserving the typestate.
// ---------------------------------------------------------------------------

macro_rules! impl_line_optional_setters {
    ($t:ty) => {
        impl<'df> $t {
            /// Column used to split rows into separate coloured series.
            ///
            /// Must be Categorical (String, Boolean, or Polars `Categorical` dtype).
            /// Null rows are collected into a synthetic `"null"` series and coloured grey.
            pub fn color_by(mut self, col: &str) -> Self {
                self.config.color_by = Some(col.to_string());
                self
            }

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

            /// Label rendered beside the y-axis. Defaults to the column name.
            pub fn y_label(mut self, label: &str) -> Self {
                self.config.y_label = Some(label.to_string());
                self
            }

            /// Visual theme. Defaults to [`Theme::Default`].
            pub fn theme(mut self, theme: Theme) -> Self {
                self.config.theme = theme;
                self
            }

            /// How null y-values are handled. Defaults to [`NullPolicy::Skip`].
            ///
            /// Both policies emit [`CharcoalWarning::NullsSkipped`] for each null y.
            pub fn null_policy(mut self, policy: NullPolicy) -> Self {
                self.config.null_policy = policy;
                self
            }

            /// Line dash style. Defaults to [`DashStyle::Solid`].
            pub fn dash_style(mut self, style: DashStyle) -> Self {
                self.config.dash_style = style;
                self
            }

            /// Stroke width in pixels. Clamped to a minimum of `0.5`. Defaults to `2.0`.
            pub fn stroke_width(mut self, px: f64) -> Self {
                self.config.stroke_width = px.max(0.5);
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

impl_line_optional_setters!(LineBuilder<'df>);
impl_line_optional_setters!(LineWithX<'df>);
impl_line_optional_setters!(LineWithXY<'df>);

// ---------------------------------------------------------------------------
// Required field transitions
// ---------------------------------------------------------------------------

impl<'df> LineBuilder<'df> {
    /// Set the x-axis column (Numeric or Temporal).
    ///
    /// Column validation is deferred to `.build()` so callers need no `?` when chaining.
    pub fn x(mut self, col: &str) -> LineWithX<'df> {
        self.config.x_col = Some(col.to_string());
        LineWithX { df: self.df, config: self.config }
    }
}

impl<'df> LineWithX<'df> {
    /// Set the y-axis column (Numeric). Unlocks `.build()`.
    ///
    /// Column validation is deferred to `.build()` so callers need no `?` when chaining.
    pub fn y(mut self, col: &str) -> LineWithXY<'df> {
        self.config.y_col = Some(col.to_string());
        LineWithXY { df: self.df, config: self.config }
    }
}

// ---------------------------------------------------------------------------
// .build()
// ---------------------------------------------------------------------------

impl<'df> LineWithXY<'df> {
    /// Validate columns, apply the null policy, render polylines, and return a [`Chart`].
    ///
    /// # Errors
    ///
    /// - [`CharcoalError::DataTooLarge`] if `df.height() > row_limit`.
    /// - [`CharcoalError::ColumnNotFound`] for any missing column name.
    /// - [`CharcoalError::UnsupportedColumn`] if x is Categorical/Unsupported, y is not
    ///   Numeric, or `color_by` is not Categorical.
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
        // 2 & 3. Classify and normalize x  (Numeric or Temporal only)
        //
        // Null-x warnings come from the normalizer — extend them here exactly
        // as scatter.rs does; do not double-count.
        // ------------------------------------------------------------------
        let x_col = config.x_col.as_deref().unwrap(); // always present (typestate)
        let x_viz  = classify_column(df, x_col, None)?;

        if x_viz == VizDtype::Categorical || x_viz == VizDtype::Unsupported {
            let dtype = df.schema().get(x_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     x_col.to_string(),
                dtype,
                message: "The x column of a line chart must be Numeric or Temporal. \
                          Categorical columns cannot be plotted on a continuous axis.".to_string(),
            });
        }

        let x_is_temporal = x_viz == VizDtype::Temporal;

        // _x_epoch_ms is kept alive so tick_labels_temporal can use the raw ms values.
        let (x_f64, _x_epoch_ms): (Vec<Option<f64>>, Option<Vec<Option<i64>>>) =
            if x_is_temporal {
                let (ms, w) = to_epoch_ms(df, x_col)?;
                warnings.extend(w); // NullsSkipped for null x, if any
                let as_f64: Vec<Option<f64>> = ms.iter().map(|v| v.map(|i| i as f64)).collect();
                (as_f64, Some(ms))
            } else {
                let (vals, w) = to_f64(df, x_col)?;
                warnings.extend(w); // NullsSkipped for null x, if any
                (vals, None)
            };

        // ------------------------------------------------------------------
        // 2 & 3. Classify and normalize y  (Numeric only)
        //
        // The normalizer would emit a NullsSkipped for the raw column name, but
        // the per-series null loop below emits richer per-label warnings instead.
        // Discard the normalizer's y-warnings here; the per-series loop handles them.
        // ------------------------------------------------------------------
        let y_col = config.y_col.as_deref().unwrap();
        let y_viz  = classify_column(df, y_col, None)?;

        if y_viz != VizDtype::Numeric {
            let dtype = df.schema().get(y_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     y_col.to_string(),
                dtype,
                message: "The y column of a line chart must be Numeric.".to_string(),
            });
        }

        let (y_vals, _y_w) = to_f64(df, y_col)?;
        // _y_w intentionally discarded — per-series loop emits the correct warnings below.

        // ------------------------------------------------------------------
        // 4. Optional color_by  (Categorical only for multi-series lines)
        // ------------------------------------------------------------------
        let color_vals: Option<Vec<Option<String>>> = match &config.color_by {
            None => None,
            Some(col) => {
                let cv = classify_column(df, col, None)?;
                if cv != VizDtype::Categorical {
                    let dtype = df.schema().get(col.as_str()).unwrap().clone();
                    return Err(CharcoalError::UnsupportedColumn {
                        col:     col.clone(),
                        dtype,
                        message: "color_by for a line chart must be a Categorical column \
                                  (String, Boolean, or Categorical dtype).".to_string(),
                    });
                }
                let (vals, w) = to_string(df, col)?;
                warnings.extend(w);
                Some(vals)
            }
        };

        // ------------------------------------------------------------------
        // 5. Build per-series point lists
        //
        // A series is a `(label, color, Vec<(x_f64, Option<y_f64>)>)` triple.
        //
        //   • Rows where x is None are always dropped here (gap regardless of
        //     NullPolicy). Their count was already captured by the normalizer
        //     warning above — no double-counting.
        //   • Null y values are preserved as `None` so the NullPolicy can act
        //     on them during rendering.
        // ------------------------------------------------------------------
        let theme_cfg = ThemeConfig::from(&config.theme);

        let series_list: Vec<(String, String, Vec<(f64, Option<f64>)>)> =
            if color_vals.is_none() {
                // ── Single series ──────────────────────────────────────────
                let color = theme_cfg.palette[0].to_string();
                let pts: Vec<(f64, Option<f64>)> = (0..n_rows)
                    .filter_map(|i| x_f64[i].map(|xv| (xv, y_vals[i])))
                    .collect();
                vec![("".to_string(), color, pts)]
            } else {
                // ── Multi-series ───────────────────────────────────────────
                // One entry per unique category in first-seen row order.
                // Null category rows become a synthetic "null" series.
                let cv = color_vals.as_ref().unwrap();

                let mut order: Vec<Option<String>> = Vec::new();
                for v in cv {
                    if !order.contains(v) {
                        order.push(v.clone());
                    }
                }

                let mut palette_idx = 0usize;
                order
                    .into_iter()
                    .map(|cat_opt| {
                        let label = cat_opt.as_deref().unwrap_or("null").to_string();
                        let color = if cat_opt.is_none() {
                            // Null category: neutral grey, not a palette colour.
                            NULL_COLOR.to_string()
                        } else {
                            let c = theme_cfg.palette[palette_idx % theme_cfg.palette.len()]
                                .to_string();
                            palette_idx += 1;
                            c
                        };
                        let pts: Vec<(f64, Option<f64>)> = (0..n_rows)
                            .filter(|&i| {
                                x_f64[i].is_some()
                                    && cv[i].as_deref() == cat_opt.as_deref()
                            })
                            .map(|i| (x_f64[i].unwrap(), y_vals[i]))
                            .collect();
                        (label, color, pts)
                    })
                    .collect()
            };

        // ------------------------------------------------------------------
        // 6. Null-y warnings  (emitted for every series, regardless of policy)
        //
        // The per-series count is taken *after* x-null rows are already dropped,
        // so this accurately reflects the null-y rows that affect the rendered line.
        // ------------------------------------------------------------------
        for (label, _, pts) in &series_list {
            let y_null_count = pts.iter().filter(|(_, y)| y.is_none()).count();
            if y_null_count > 0 {
                // For single series `label` is ""; use the raw column name.
                let col_label = if label.is_empty() {
                    y_col.to_string()
                } else {
                    format!("{} ({})", y_col, label)
                };
                warnings.push(CharcoalWarning::NullsSkipped {
                    col:   col_label,
                    count: y_null_count,
                });
            }
        }

        // ------------------------------------------------------------------
        // 7. Axis ranges from all non-null values across all series
        // ------------------------------------------------------------------
        let all_x: Vec<f64> = series_list
            .iter()
            .flat_map(|(_, _, pts)| pts.iter().map(|(x, _)| *x))
            .collect();
        let all_y: Vec<f64> = series_list
            .iter()
            .flat_map(|(_, _, pts)| pts.iter().filter_map(|(_, y)| *y))
            .collect();

        let x_min = all_x.iter().cloned().fold(f64::INFINITY,     f64::min);
        let x_max = all_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = all_y.iter().cloned().fold(f64::INFINITY,     f64::min);
        let y_max = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Fall back to a [0, 1] placeholder when the dataset is empty or all-null.
        let (x_min, x_max) = if x_min.is_infinite() { (0.0, 1.0) } else { (x_min, x_max) };
        let (y_min, y_max) = if y_min.is_infinite() { (0.0, 1.0) } else { (y_min, y_max) };

        let x_tick_vals = nice_ticks(x_min, x_max, 6);
        let y_tick_vals = nice_ticks(y_min, y_max, 6);

        // ------------------------------------------------------------------
        // 8. Canvas and scales
        // ------------------------------------------------------------------
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

        // x: data left → pixel right
        // y: SVG y increases downward, so data_min maps to pixel bottom.
        let x_scale = LinearScale::new(
            *x_tick_vals.first().unwrap(),
            *x_tick_vals.last().unwrap(),
            ox,
            ox + pw,
        );
        let y_scale = LinearScale::new(
            *y_tick_vals.first().unwrap(),
            *y_tick_vals.last().unwrap(),
            oy + ph, // data min → pixel bottom
            oy,      // data max → pixel top
        );

        // ------------------------------------------------------------------
        // 9. Render polylines — apply NullPolicy per series
        //
        // geometry::polyline() uses <path d="M…L…"> notation (not <polyline>).
        // The SVG element tag is <path>; tests count "<path" occurrences.
        // ------------------------------------------------------------------
        let stroke_w = config.stroke_width;
        // DashStyle::stroke_dasharray() → Option<&'static str>.
        // geometry::polyline() takes &str where "" means no dasharray attribute.
        let dash = config.dash_style.stroke_dasharray().unwrap_or("");

        let mut elements: Vec<String>                = Vec::new();
        let mut legend_entries: Vec<(String, String)> = Vec::new();

        for (label, color, pts) in &series_list {
            let segments: Vec<Vec<(f64, f64)>> = match config.null_policy {
                NullPolicy::Skip => {
                    segments_skip(pts, &x_scale, &y_scale)
                }
                NullPolicy::Interpolate => {
                    let filled = interpolate_nulls(pts);
                    segments_skip(&filled, &x_scale, &y_scale)
                }
            };

            for seg in segments {
                // A line requires at least 2 endpoints; silently drop single-point segments.
                if seg.len() >= 2 {
                    elements.push(geometry::polyline(&seg, color, stroke_w, dash));
                }
            }

            if config.color_by.is_some() {
                legend_entries.push((label.clone(), color.clone()));
            }
        }

        // ------------------------------------------------------------------
        // 10. Axis SVG
        // ------------------------------------------------------------------
        let x_labels = if x_is_temporal {
            let range_ms = (x_max - x_min) as i64;
            let x_tick_i64: Vec<i64> = x_tick_vals.iter().map(|&v| v as i64).collect();
            tick_labels_temporal(&x_tick_i64, range_ms)
        } else {
            tick_labels_numeric(&x_tick_vals)
        };
        let y_labels = tick_labels_numeric(&y_tick_vals);

        let x_ticks = build_tick_marks(&x_tick_vals, &x_labels, &x_scale);
        let y_ticks = build_tick_marks(&y_tick_vals, &y_labels, &y_scale);

        let x_axis = compute_axis(
            &x_scale, &x_ticks, AxisOrientation::Horizontal,
            ox, oy, pw, ph, &theme_cfg,
        );
        let y_axis = compute_axis(
            &y_scale, &y_ticks, AxisOrientation::Vertical,
            ox, oy, pw, ph, &theme_cfg,
        );

        // ------------------------------------------------------------------
        // 11. Legend  (only present when color_by is set)
        // ------------------------------------------------------------------
        let legend: Option<Vec<(String, String)>> = if legend_entries.is_empty() {
            None
        } else {
            Some(legend_entries)
        };

        // ------------------------------------------------------------------
        // 12. Assemble Chart
        // ------------------------------------------------------------------
        let title   = config.title.as_deref().unwrap_or("");
        let x_label = config.x_label.as_deref().unwrap_or(x_col);
        let y_label = config.y_label.as_deref().unwrap_or(y_col);

        let svg = canvas.render(elements, x_axis, y_axis, title, x_label, y_label, legend);

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
// Null-handling helpers (free functions, not methods — Area will reuse them)
// ---------------------------------------------------------------------------

/// Splits a series of `(x, Option<y>)` data-space points into contiguous non-null
/// pixel-space segments.
///
/// A `None` y ends the current segment; the next `Some` y begins a new one.
/// Single-point segments are **kept** here and filtered to `len >= 2` by the caller.
pub(crate) fn segments_skip(
    pts:     &[(f64, Option<f64>)],
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) -> Vec<Vec<(f64, f64)>> {
    let mut result:  Vec<Vec<(f64, f64)>> = Vec::new();
    let mut current: Vec<(f64, f64)>       = Vec::new();

    for &(x, y_opt) in pts {
        match y_opt {
            Some(y) => current.push((x_scale.map(x), y_scale.map(y))),
            None => {
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
            }
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Fills *interior* null y-values by linear interpolation keyed on x-coordinate distance.
///
/// Leading and trailing nulls (no non-null neighbour on one side) are left as `None`
/// so they become gaps in the subsequent [`segments_skip`] pass — the spec explicitly
/// forbids extrapolation.
pub(crate) fn interpolate_nulls(pts: &[(f64, Option<f64>)]) -> Vec<(f64, Option<f64>)> {
    let n = pts.len();
    let mut result: Vec<(f64, Option<f64>)> = pts.to_vec();

    let mut i = 0;
    while i < n {
        if result[i].1.is_none() {
            // Advance to find the end of this null run.
            let null_start = i;
            while i < n && result[i].1.is_none() {
                i += 1;
            }
            let null_end = i; // exclusive — first index after the run

            let left  = if null_start > 0  { result[null_start - 1].1 } else { None };
            let right = if null_end   < n  { result[null_end].1       } else { None };

            if let (Some(lv), Some(rv)) = (left, right) {
                // Both neighbours present — fill by x-distance-weighted lerp.
                let lx     = result[null_start - 1].0;
                let rx     = result[null_end].0;
                let x_span = rx - lx;

                for j in null_start..null_end {
                    let frac = if x_span == 0.0 {
                        0.5 // degenerate: all x equal → place at midpoint
                    } else {
                        (result[j].0 - lx) / x_span
                    };
                    result[j].1 = Some(lv + frac * (rv - lv));
                }
            }
            // Edge nulls: left or right missing → leave as None (gap, not extrapolation).
        } else {
            i += 1;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod setter_tests {
    use super::*;
    use polars::frame::DataFrame;

    fn empty_df() -> DataFrame { DataFrame::empty() }

    #[test]
    fn color_by_stores_col() {
        let df = empty_df();
        let b = LineBuilder::new(&df).color_by("series");
        assert_eq!(b.config.color_by.as_deref(), Some("series"));
    }

    #[test]
    fn title_stores_string() {
        let df = empty_df();
        assert_eq!(LineBuilder::new(&df).title("T").config.title.as_deref(), Some("T"));
    }

    #[test]
    fn x_label_stores_string() {
        let df = empty_df();
        assert_eq!(LineBuilder::new(&df).x_label("X").config.x_label.as_deref(), Some("X"));
    }

    #[test]
    fn y_label_stores_string() {
        let df = empty_df();
        assert_eq!(LineBuilder::new(&df).y_label("Y").config.y_label.as_deref(), Some("Y"));
    }

    #[test]
    fn theme_stores_value() {
        let df = empty_df();
        assert!(matches!(LineBuilder::new(&df).theme(Theme::Dark).config.theme, Theme::Dark));
    }

    #[test]
    fn null_policy_stores_interpolate() {
        let df = empty_df();
        assert_eq!(
            LineBuilder::new(&df).null_policy(NullPolicy::Interpolate).config.null_policy,
            NullPolicy::Interpolate,
        );
    }

    #[test]
    fn dash_style_stores_dashed() {
        let df = empty_df();
        assert_eq!(
            LineBuilder::new(&df).dash_style(DashStyle::Dashed).config.dash_style,
            DashStyle::Dashed,
        );
    }

    #[test]
    fn stroke_width_stores_value() {
        let df = empty_df();
        let b = LineBuilder::new(&df).stroke_width(3.0);
        assert!((b.config.stroke_width - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stroke_width_clamps_to_minimum() {
        let df = empty_df();
        assert!(LineBuilder::new(&df).stroke_width(0.0).config.stroke_width >= 0.5);
    }

    #[test]
    fn row_limit_stores_value() {
        let df = empty_df();
        assert_eq!(LineBuilder::new(&df).row_limit(500_000).config.row_limit, 500_000);
    }

    #[test]
    fn defaults_are_sane() {
        let df = empty_df();
        let b = LineBuilder::new(&df);
        assert!(b.config.color_by.is_none());
        assert!(b.config.title.is_none());
        assert!(matches!(b.config.theme, Theme::Default));
        assert_eq!(b.config.null_policy, NullPolicy::Skip);
        assert_eq!(b.config.dash_style,  DashStyle::Solid);
        assert!((b.config.stroke_width - DEFAULT_STROKE).abs() < f64::EPSILON);
        assert_eq!(b.config.row_limit, DEFAULT_ROW_LIMIT);
    }

    #[test]
    fn setters_available_on_line_with_x() {
        let df = empty_df();
        let s = LineWithX {
            df: &df,
            config: LineConfig { x_col: Some("t".to_string()), ..Default::default() },
        };
        let s = s.null_policy(NullPolicy::Interpolate).dash_style(DashStyle::Dotted);
        assert_eq!(s.config.null_policy, NullPolicy::Interpolate);
        assert_eq!(s.config.dash_style,  DashStyle::Dotted);
    }

    #[test]
    fn setters_available_on_line_with_xy() {
        let df = empty_df();
        let s = LineWithXY {
            df: &df,
            config: LineConfig {
                x_col: Some("t".to_string()),
                y_col: Some("v".to_string()),
                ..Default::default()
            },
        };
        let s = s.theme(Theme::Colorblind).stroke_width(4.0).row_limit(200_000);
        assert!(matches!(s.config.theme, Theme::Colorblind));
        assert!((s.config.stroke_width - 4.0).abs() < f64::EPSILON);
        assert_eq!(s.config.row_limit, 200_000);
    }

    #[test]
    fn chained_setters_preserve_df_pointer() {
        let df = empty_df();
        let b = LineBuilder::new(&df)
            .color_by("c")
            .title("T")
            .theme(Theme::Minimal)
            .null_policy(NullPolicy::Interpolate)
            .dash_style(DashStyle::Dashed)
            .stroke_width(2.5)
            .row_limit(100_000);
        assert!(std::ptr::eq(b.df, &df));
    }
}

#[cfg(test)]
mod transition_tests {
    use super::*;
    use polars::frame::DataFrame;

    fn empty_df() -> DataFrame { DataFrame::empty() }

    #[test]
    fn x_stores_col_and_transitions() {
        let df = empty_df();
        assert_eq!(LineBuilder::new(&df).x("time").config.x_col.as_deref(), Some("time"));
    }

    #[test]
    fn y_stores_col_and_unlocks_build() {
        let df = empty_df();
        let b = LineBuilder::new(&df).x("time").y("value");
        assert_eq!(b.config.x_col.as_deref(), Some("time"));
        assert_eq!(b.config.y_col.as_deref(), Some("value"));
    }

    #[test]
    fn optional_setters_survive_both_transitions() {
        let df = empty_df();
        let b = LineBuilder::new(&df)
            .title("Before X")
            .x("t")
            .null_policy(NullPolicy::Interpolate)
            .y("v")
            .dash_style(DashStyle::Dashed);
        assert_eq!(b.config.title.as_deref(), Some("Before X"));
        assert_eq!(b.config.null_policy, NullPolicy::Interpolate);
        assert_eq!(b.config.dash_style,  DashStyle::Dashed);
    }
}

#[cfg(test)]
mod build_tests {
    use super::*;
    use polars::prelude::*;

    // ── Fixtures ──────────────────────────────────────────────────────────

    /// Five-point clean series with no nulls.
    fn timeseries_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("t", &[1.0f64, 2.0, 3.0, 4.0, 5.0]),
            Series::new("v", &[10.0f64, 20.0, 15.0, 25.0, 18.0]),
        ]).unwrap()
    }

    /// Interior null at index 1: [10, NULL, 15, 25, 18].
    /// Left flank = [10]  (1 point → dropped by caller).
    /// Right flank = [15, 25, 18]  (3 points → kept).
    fn df_null_y_interior() -> DataFrame {
        DataFrame::new(vec![
            Series::new("t", &[Some(1.0f64), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]),
            Series::new("v", &[Some(10.0f64), None, Some(15.0), Some(25.0), Some(18.0)]),
        ]).unwrap()
    }

    /// Interior null at index 2: [10, 20, NULL, 25, 18].
    /// Left flank = [10, 20]  (2 points → kept).
    /// Right flank = [25, 18]  (2 points → kept).
    /// → Skip produces **two** <path> elements.
    fn df_null_y_two_flanks() -> DataFrame {
        DataFrame::new(vec![
            Series::new("t", &[Some(1.0f64), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]),
            Series::new("v", &[Some(10.0f64), Some(20.0), None, Some(25.0), Some(18.0)]),
        ]).unwrap()
    }

    /// Leading null at index 0: [NULL, 20, 15].
    /// No left neighbour → stays a gap even under Interpolate.
    fn df_null_y_leading() -> DataFrame {
        DataFrame::new(vec![
            Series::new("t", &[Some(1.0f64), Some(2.0), Some(3.0)]),
            Series::new("v", &[None, Some(20.0), Some(15.0)]),
        ]).unwrap()
    }

    /// One null x at index 1.
    fn df_null_x() -> DataFrame {
        DataFrame::new(vec![
            Series::new("t", &[Some(1.0f64), None, Some(3.0), Some(4.0), Some(5.0)]),
            Series::new("v", &[Some(10.0f64), Some(20.0), Some(15.0), Some(25.0), Some(18.0)]),
        ]).unwrap()
    }

    /// Two series, A and B, interleaved.  Each series has 3 contiguous points.
    fn multiseries_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("t",      &[1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0]),
            Series::new("v",      &[10.0f64, 20.0, 15.0, 5.0, 12.0, 8.0]),
            Series::new("series", &["A", "A", "A", "B", "B", "B"]),
        ]).unwrap()
    }

    // ── Happy path ────────────────────────────────────────────────────────

    #[test]
    fn clean_series_produces_valid_svg_with_path_element() {
        let df = timeseries_df();
        let chart = LineBuilder::new(&df)
            .x("t").y("v")
            .build()
            .expect("clean timeseries must build");
        assert!(chart.svg().contains("<svg"),  "output must be SVG");
        // geometry::polyline renders as <path d="M…L…"> not <polyline>.
        assert!(chart.svg().contains("<path"), "line chart must contain <path> elements");
    }

    #[test]
    fn build_produces_correct_dimensions() {
        let df = timeseries_df();
        let chart = LineBuilder::new(&df).x("t").y("v").build().unwrap();
        assert_eq!(chart.width(),  CANVAS_WIDTH);
        assert_eq!(chart.height(), CANVAS_HEIGHT);
    }

    #[test]
    fn clean_data_emits_no_warnings() {
        let df = timeseries_df();
        let chart = LineBuilder::new(&df).x("t").y("v").build().unwrap();
        assert!(chart.warnings().is_empty(), "clean data must produce no warnings");
    }

    #[test]
    fn single_row_does_not_panic() {
        let df = DataFrame::new(vec![
            Series::new("t", &[1.0f64]),
            Series::new("v", &[5.0f64]),
        ]).unwrap();
        LineBuilder::new(&df).x("t").y("v").build()
            .expect("single-row dataset must build");
    }

    // ── NullPolicy::Skip ─────────────────────────────────────────────────

    #[test]
    fn skip_emits_null_y_warning() {
        let df = df_null_y_interior();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .null_policy(NullPolicy::Skip)
            .build().unwrap();
        let warned = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, .. } if col.starts_with('v')
        ));
        assert!(warned, "Skip must emit NullsSkipped for null y");
    }

    #[test]
    fn skip_with_two_valid_flanks_produces_two_path_elements() {
        // [10, 20, NULL, 25, 18]: left=[10,20]≥2, right=[25,18]≥2 → 2 <path>
        let df = df_null_y_two_flanks();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .null_policy(NullPolicy::Skip)
            .build().unwrap();
        let path_count = chart.svg().matches("<path").count();
        assert_eq!(path_count, 2,
            "Skip with two ≥2-point flanks must produce 2 <path> elements; got {path_count}");
    }

    #[test]
    fn skip_single_point_left_flank_is_silently_dropped() {
        // [10, NULL, 15, 25, 18]: left=[10]→1pt dropped, right=[15,25,18]→kept → 1 <path>
        let df = df_null_y_interior();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .null_policy(NullPolicy::Skip)
            .build().unwrap();
        let path_count = chart.svg().matches("<path").count();
        assert_eq!(path_count, 1,
            "1-point left segment must be dropped; got {path_count}");
    }

    // ── NullPolicy::Interpolate ───────────────────────────────────────────

    #[test]
    fn interpolate_interior_null_produces_single_continuous_path() {
        // [10, NULL, 15, 25, 18]: null filled → all 5 pts form one segment → 1 <path>
        let df = df_null_y_interior();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .null_policy(NullPolicy::Interpolate)
            .build().unwrap();
        let path_count = chart.svg().matches("<path").count();
        assert_eq!(path_count, 1,
            "Interpolate on interior null must produce 1 <path>; got {path_count}");
    }

    #[test]
    fn interpolate_still_emits_null_y_warning() {
        let df = df_null_y_interior();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .null_policy(NullPolicy::Interpolate)
            .build().unwrap();
        let warned = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, .. } if col.starts_with('v')
        ));
        assert!(warned,
            "Interpolate must still emit NullsSkipped even when the gap is filled");
    }

    #[test]
    fn interpolate_leading_null_is_not_extrapolated() {
        // [NULL, 20, 15]: no left neighbour → stays gap → segment [20, 15] → 1 <path>
        let df = df_null_y_leading();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .null_policy(NullPolicy::Interpolate)
            .build().unwrap();
        let path_count = chart.svg().matches("<path").count();
        assert_eq!(path_count, 1,
            "Leading null must remain a gap, not be extrapolated; got {path_count} paths");
    }

    // ── Null x  (always a gap, regardless of NullPolicy) ─────────────────

    #[test]
    fn null_x_emits_warning_under_skip() {
        let df = df_null_x();
        let chart = LineBuilder::new(&df).x("t").y("v").build().unwrap();
        let warned = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, .. } if col == "t"
        ));
        assert!(warned, "null x must emit NullsSkipped for the x column under Skip");
    }

    #[test]
    fn null_x_emits_warning_under_interpolate() {
        let df = df_null_x();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .null_policy(NullPolicy::Interpolate)
            .build().unwrap();
        let warned = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, .. } if col == "t"
        ));
        assert!(warned, "null x must emit NullsSkipped under Interpolate too");
    }

    #[test]
    fn null_x_warning_is_not_doubled() {
        // The normalizer emits exactly one NullsSkipped for the x column.
        // build() must not push a second one on top of it.
        let df = df_null_x();
        let chart = LineBuilder::new(&df).x("t").y("v").build().unwrap();
        let x_warning_count = chart.warnings().iter()
            .filter(|w| matches!(w, CharcoalWarning::NullsSkipped { col, .. } if col == "t"))
            .count();
        assert_eq!(x_warning_count, 1, "null-x warning must appear exactly once; got {x_warning_count}");
    }

    // ── Multi-series (color_by) ───────────────────────────────────────────

    #[test]
    fn color_by_produces_one_path_per_series() {
        let df = multiseries_df();
        let chart = LineBuilder::new(&df)
            .x("t").y("v").color_by("series")
            .build().unwrap();
        // Series A: [A1,A2,A3]  →  1 segment (3 pts ≥ 2)  →  1 <path>
        // Series B: [B1,B2,B3]  →  1 segment (3 pts ≥ 2)  →  1 <path>
        // Total: 2 <path> elements.
        let path_count = chart.svg().matches("<path").count();
        assert_eq!(path_count, 2,
            "two series must produce 2 <path> elements; got {path_count}");
    }

    #[test]
    fn color_by_legend_contains_both_labels() {
        let df = multiseries_df();
        let svg = LineBuilder::new(&df)
            .x("t").y("v").color_by("series")
            .build().unwrap()
            .svg().to_string();
        assert!(svg.contains("A"), "legend must contain series label 'A'");
        assert!(svg.contains("B"), "legend must contain series label 'B'");
    }

    #[test]
    fn null_category_is_labelled_null_and_coloured_grey() {
        let df = DataFrame::new(vec![
            Series::new("t",   &[1.0f64, 2.0, 3.0]),
            Series::new("v",   &[10.0f64, 20.0, 15.0]),
            Series::new("cat", &[Some("A"), None, Some("A")]),
        ]).unwrap();
        let chart = LineBuilder::new(&df)
            .x("t").y("v").color_by("cat")
            .build().unwrap();
        assert!(chart.svg().contains("null"),
            "null category must appear as 'null' in the legend");
        assert!(chart.svg().contains(NULL_COLOR),
            "null category must use NULL_COLOR ({NULL_COLOR}), not a palette colour");
    }

    #[test]
    fn palette_cycles_when_categories_exceed_palette_length() {
        // Default palette has 8 entries.  Build 9 distinct categories.
        let cats: Vec<&str> = vec!["a","b","c","d","e","f","g","h","i"];
        let xs:   Vec<f64>  = (0..9).map(|i| i as f64).collect();
        let ys:   Vec<f64>  = xs.clone();
        let df = DataFrame::new(vec![
            Series::new("x",   &xs),
            Series::new("y",   &ys),
            Series::new("cat", &cats),
        ]).unwrap();
        // Must not panic; all 9 labels must appear in the legend.
        let chart = LineBuilder::new(&df)
            .x("x").y("y").color_by("cat")
            .build().unwrap();
        for label in &cats {
            assert!(chart.svg().contains(label),
                "legend must contain category '{label}'");
        }
    }

    // ── Dash style ────────────────────────────────────────────────────────

    #[test]
    fn solid_produces_no_stroke_dasharray_attribute() {
        let df = timeseries_df();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .dash_style(DashStyle::Solid)
            .build().unwrap();
        assert!(!chart.svg().contains("stroke-dasharray"),
            "Solid must not produce a stroke-dasharray attribute");
    }

    #[test]
    fn dashed_produces_stroke_dasharray_6_3() {
        let df = timeseries_df();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .dash_style(DashStyle::Dashed)
            .build().unwrap();
        assert!(chart.svg().contains(r#"stroke-dasharray="6 3""#),
            "Dashed must produce stroke-dasharray=\"6 3\"");
    }

    #[test]
    fn dotted_produces_stroke_dasharray_2_2() {
        let df = timeseries_df();
        let chart = LineBuilder::new(&df).x("t").y("v")
            .dash_style(DashStyle::Dotted)
            .build().unwrap();
        assert!(chart.svg().contains(r#"stroke-dasharray="2 2""#),
            "Dotted must produce stroke-dasharray=\"2 2\"");
    }

    // ── Error paths ───────────────────────────────────────────────────────

    #[test]
    fn x_typo_returns_column_not_found() {
        let df = timeseries_df();
        let err = LineBuilder::new(&df).x("tmie").y("v").build().unwrap_err();
        assert!(matches!(err, CharcoalError::ColumnNotFound { .. }),
            "x typo must return ColumnNotFound");
    }

    #[test]
    fn y_typo_returns_column_not_found() {
        let df = timeseries_df();
        let err = LineBuilder::new(&df).x("t").y("vlaue").build().unwrap_err();
        assert!(matches!(err, CharcoalError::ColumnNotFound { .. }),
            "y typo must return ColumnNotFound");
    }

    #[test]
    fn categorical_x_returns_unsupported_column() {
        let df = DataFrame::new(vec![
            Series::new("cat", &["a", "b", "c"]),
            Series::new("v",   &[1.0f64, 2.0, 3.0]),
        ]).unwrap();
        let err = LineBuilder::new(&df).x("cat").y("v").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { .. }),
            "categorical x must return UnsupportedColumn");
    }

    #[test]
    fn categorical_y_returns_unsupported_column() {
        let df = multiseries_df();
        let err = LineBuilder::new(&df).x("t").y("series").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { .. }),
            "categorical y must return UnsupportedColumn");
    }

    #[test]
    fn row_limit_exceeded_returns_data_too_large() {
        let df = timeseries_df(); // 5 rows
        let err = LineBuilder::new(&df).x("t").y("v").row_limit(3).build().unwrap_err();
        match err {
            CharcoalError::DataTooLarge { rows, limit, .. } => {
                assert_eq!(rows, 5);
                assert_eq!(limit, 3);
            }
            other => panic!("expected DataTooLarge, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests for the two helper functions (independent of build())
// ---------------------------------------------------------------------------

#[cfg(test)]
mod interpolate_unit_tests {
    use super::*;

    #[test]
    fn single_interior_null_at_even_midpoint() {
        // x:[0,1,2]  y:[0,NULL,2]  →  x=1 is midpoint  →  y=1
        let pts = vec![(0.0_f64, Some(0.0_f64)), (1.0, None), (2.0, Some(2.0))];
        let out = interpolate_nulls(&pts);
        let v = out[1].1.expect("interior null must be filled");
        assert!((v - 1.0).abs() < 1e-9, "midpoint must yield 1.0; got {v}");
    }

    #[test]
    fn consecutive_interior_nulls_filled_correctly() {
        // x:[0,1,2,3]  y:[0,NULL,NULL,3]
        // frac(x=1) = 1/3 → y=1.0  |  frac(x=2) = 2/3 → y=2.0
        let pts = vec![
            (0.0_f64, Some(0.0_f64)),
            (1.0,     None),
            (2.0,     None),
            (3.0,     Some(3.0)),
        ];
        let out = interpolate_nulls(&pts);
        let v1 = out[1].1.expect("first null must be filled");
        let v2 = out[2].1.expect("second null must be filled");
        assert!((v1 - 1.0).abs() < 1e-9, "first gap: expected 1.0; got {v1}");
        assert!((v2 - 2.0).abs() < 1e-9, "second gap: expected 2.0; got {v2}");
    }

    #[test]
    fn leading_null_left_as_gap() {
        let pts = vec![(0.0_f64, None), (1.0, Some(5.0_f64)), (2.0, Some(10.0))];
        assert!(interpolate_nulls(&pts)[0].1.is_none(), "leading null must remain None");
    }

    #[test]
    fn trailing_null_left_as_gap() {
        let pts = vec![(0.0_f64, Some(0.0_f64)), (1.0, Some(5.0)), (2.0, None)];
        assert!(interpolate_nulls(&pts)[2].1.is_none(), "trailing null must remain None");
    }

    #[test]
    fn uneven_x_spacing_uses_x_distance_not_index() {
        // x:[0,1,9]  y:[0,NULL,9]
        // frac = (1-0)/(9-0) = 1/9  →  y = 0 + (1/9)*9 = 1.0  (not 4.5 which index-based would give)
        let pts = vec![(0.0_f64, Some(0.0_f64)), (1.0, None), (9.0, Some(9.0))];
        let v = interpolate_nulls(&pts)[1].1.expect("null must be filled");
        assert!((v - 1.0).abs() < 1e-9, "x-distance interpolation: expected 1.0; got {v}");
    }

    #[test]
    fn all_nulls_remain_as_gaps() {
        let pts: Vec<(f64, Option<f64>)> = vec![(0.0, None), (1.0, None), (2.0, None)];
        for (i, (_, y)) in interpolate_nulls(&pts).iter().enumerate() {
            assert!(y.is_none(), "index {i}: all-null must remain None");
        }
    }

    #[test]
    fn no_nulls_returns_values_unchanged() {
        let pts = vec![(0.0_f64, Some(1.0_f64)), (1.0, Some(2.0)), (2.0, Some(3.0))];
        let out = interpolate_nulls(&pts);
        for (i, ((_, oy), (_, ry))) in pts.iter().zip(out.iter()).enumerate() {
            assert_eq!(*oy, *ry, "index {i}: non-null must not change");
        }
    }
}

#[cfg(test)]
mod segments_skip_unit_tests {
    use super::*;
    use crate::render::axes::LinearScale;

    fn identity_scale() -> LinearScale {
        // 1:1 mapping so pixel coords equal data coords — simplifies assertions.
        LinearScale::new(0.0, 100.0, 0.0, 100.0)
    }

    #[test]
    fn no_nulls_returns_single_segment_with_all_points() {
        let xs = identity_scale();
        let ys = identity_scale();
        let pts = vec![(1.0, Some(1.0)), (2.0, Some(2.0)), (3.0, Some(3.0))];
        let segs = segments_skip(&pts, &xs, &ys);
        assert_eq!(segs.len(), 1, "no nulls → 1 segment");
        assert_eq!(segs[0].len(), 3);
    }

    #[test]
    fn null_in_middle_splits_into_two_segments() {
        let xs = identity_scale();
        let ys = identity_scale();
        let pts = vec![(1.0, Some(1.0)), (2.0, None), (3.0, Some(3.0))];
        let segs = segments_skip(&pts, &xs, &ys);
        assert_eq!(segs.len(), 2, "interior null → 2 segments");
        assert_eq!(segs[0].len(), 1); // [1]
        assert_eq!(segs[1].len(), 1); // [3]
    }

    #[test]
    fn leading_null_produces_one_segment_from_rest() {
        let xs = identity_scale();
        let ys = identity_scale();
        let pts = vec![(1.0, None), (2.0, Some(2.0)), (3.0, Some(3.0))];
        let segs = segments_skip(&pts, &xs, &ys);
        assert_eq!(segs.len(), 1, "leading null → 1 segment");
        assert_eq!(segs[0].len(), 2);
    }

    #[test]
    fn trailing_null_produces_one_segment_from_rest() {
        let xs = identity_scale();
        let ys = identity_scale();
        let pts = vec![(1.0, Some(1.0)), (2.0, Some(2.0)), (3.0, None)];
        let segs = segments_skip(&pts, &xs, &ys);
        assert_eq!(segs.len(), 1, "trailing null → 1 segment");
        assert_eq!(segs[0].len(), 2);
    }

    #[test]
    fn all_nulls_returns_empty() {
        let xs = identity_scale();
        let ys = identity_scale();
        let pts: Vec<(f64, Option<f64>)> = vec![(1.0, None), (2.0, None)];
        assert!(segments_skip(&pts, &xs, &ys).is_empty(), "all nulls → 0 segments");
    }

    #[test]
    fn empty_input_returns_empty() {
        let xs = identity_scale();
        let ys = identity_scale();
        assert!(segments_skip(&[], &xs, &ys).is_empty());
    }
}