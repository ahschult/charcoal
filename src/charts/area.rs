//! Area chart builder (`Chart::area(&df)`).
//!
//! Area charts are structurally similar to line charts, but fill the region between the
//! plotted line and a baseline. The key additions over Line are:
//!
//! - **[`FillMode`]** — controls what the baseline is:
//!   - `ToZero` (default) fills down to y=0.
//!   - `ToMinimum` fills down to the minimum y value in the data.
//!   - `Between` fills the region between exactly two series (requires `color_by`).
//! - **Stacking** — `.stacked(true)` with `color_by` stacks series on top of one another,
//!   plotting each on a cumulative baseline.
//! - **[`NullPolicy`]** and **[`DashStyle`]** are inherited from Line.
//!
//! # Null semantics
//!
//! | Column              | Behaviour                                                      |
//! |---------------------|----------------------------------------------------------------|
//! | `x` (any)           | Row dropped; `NullsSkipped` warning emitted.                   |
//! | `y` under `Skip`    | Filled region breaks at nulls; multiple polygons rendered.     |
//! | `y` under `Interpolate` | Same interpolation as Line before polygon construction.    |
//!
//! A [`CharcoalWarning::NullsSkipped`] warning is emitted for every null y value regardless
//! of `NullPolicy`.

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

// Re-use the null-handling helpers from line.rs — they are pub(crate).
use crate::charts::line::{interpolate_nulls, segments_skip};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CANVAS_WIDTH:      u32   = 800;
const CANVAS_HEIGHT:     u32   = 500;
const DEFAULT_STROKE:    f64   = 1.5;
const DEFAULT_ROW_LIMIT: usize = 1_000_000;
/// Default fill opacity — lower than 1.0 so stacked layers show through.
const FILL_OPACITY: f64 = 0.6;
/// Neutral grey for the synthetic null-category series.
const NULL_COLOR: &str = "#AAAAAA";

// ---------------------------------------------------------------------------
// Public enums
// ---------------------------------------------------------------------------

/// Controls the baseline of the filled area.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillMode {
    /// Fill from the line down to y=0. Default.
    #[default]
    ToZero,
    /// Fill from the line down to the minimum y value in the data.
    ToMinimum,
    /// Fill the region between exactly two series.
    ///
    /// Requires `color_by` with exactly two unique category values.
    /// Returns [`CharcoalError::InsufficientData`] if there are not exactly 2 series.
    Between,
}

// ---------------------------------------------------------------------------
// Accumulated configuration
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct AreaConfig {
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
    pub fill_mode:    FillMode,
    pub stacked:      bool,
    pub row_limit:    usize,
}

impl Default for AreaConfig {
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
            fill_mode:    FillMode::ToZero,
            stacked:      false,
            row_limit:    DEFAULT_ROW_LIMIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder states (typestate pattern)
// ---------------------------------------------------------------------------

/// Initial area builder — no required fields set yet.
pub struct AreaBuilder<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: AreaConfig,
}

/// x-axis column set; y still required.
pub struct AreaWithX<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: AreaConfig,
}

/// Both required fields set — `.build()` is available.
pub struct AreaWithXY<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: AreaConfig,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'df> AreaBuilder<'df> {
    pub(crate) fn new(df: &'df DataFrame) -> Self {
        Self { df, config: AreaConfig::default() }
    }
}

// ---------------------------------------------------------------------------
// Optional setters — applied to all three states via macro
// ---------------------------------------------------------------------------

macro_rules! impl_area_optional_setters {
    ($t:ty) => {
        impl<'df> $t {
            /// Column used to split rows into separate coloured series.
            ///
            /// Must be Categorical. Required for [`FillMode::Between`] (exactly 2 series).
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
            pub fn null_policy(mut self, policy: NullPolicy) -> Self {
                self.config.null_policy = policy;
                self
            }

            /// Stroke style for the top edge of each area. Defaults to [`DashStyle::Solid`].
            pub fn dash_style(mut self, style: DashStyle) -> Self {
                self.config.dash_style = style;
                self
            }

            /// Stroke width in pixels for the top edge. Clamped to minimum 0.5.
            pub fn stroke_width(mut self, px: f64) -> Self {
                self.config.stroke_width = px.max(0.5);
                self
            }

            /// Controls the baseline of the filled area. Defaults to [`FillMode::ToZero`].
            pub fn fill_mode(mut self, mode: FillMode) -> Self {
                self.config.fill_mode = mode;
                self
            }

            /// Stack multiple series on top of each other (requires `color_by`).
            ///
            /// Each series is plotted on top of the cumulative sum of all previous series.
            pub fn stacked(mut self, yes: bool) -> Self {
                self.config.stacked = yes;
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

impl_area_optional_setters!(AreaBuilder<'df>);
impl_area_optional_setters!(AreaWithX<'df>);
impl_area_optional_setters!(AreaWithXY<'df>);

// ---------------------------------------------------------------------------
// Required-field transitions
// ---------------------------------------------------------------------------

impl<'df> AreaBuilder<'df> {
    /// Set the x-axis column (Numeric or Temporal).
    pub fn x(mut self, col: &str) -> AreaWithX<'df> {
        self.config.x_col = Some(col.to_string());
        AreaWithX { df: self.df, config: self.config }
    }
}

impl<'df> AreaWithX<'df> {
    /// Set the y-axis column (must be Numeric).
    pub fn y(mut self, col: &str) -> AreaWithXY<'df> {
        self.config.y_col = Some(col.to_string());
        AreaWithXY { df: self.df, config: self.config }
    }
}

// ---------------------------------------------------------------------------
// .build()
// ---------------------------------------------------------------------------

impl<'df> AreaWithXY<'df> {
    /// Validate inputs, compute scales, render polygons and return a [`Chart`].
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
                message: "Reduce the dataset or raise `.row_limit()`.".to_string(),
            });
        }

        // ------------------------------------------------------------------
        // 2. Validate and normalise the x column
        // ------------------------------------------------------------------
        let x_col = config.x_col.as_deref().unwrap();
        let x_viz = classify_column(df, x_col, None)?;

        if x_viz == VizDtype::Categorical || x_viz == VizDtype::Unsupported {
            let dtype = df.schema().get(x_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     x_col.to_string(),
                dtype,
                message: "The x column of an area chart must be Numeric or Temporal.".to_string(),
            });
        }

        let x_is_temporal = x_viz == VizDtype::Temporal;
        let x_f64: Vec<Option<f64>>;
        if x_is_temporal {
            let (epoch_vals, x_w) = to_epoch_ms(df, x_col)?;
            warnings.extend(x_w);
            x_f64 = epoch_vals.into_iter().map(|v| v.map(|ms| ms as f64)).collect();
        } else {
            let (vals, x_w) = to_f64(df, x_col)?;
            warnings.extend(x_w);
            x_f64 = vals;
        };

        // ------------------------------------------------------------------
        // 3. Validate and normalise the y column
        // ------------------------------------------------------------------
        let y_col = config.y_col.as_deref().unwrap();
        let y_viz = classify_column(df, y_col, None)?;

        if y_viz != VizDtype::Numeric {
            let dtype = df.schema().get(y_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     y_col.to_string(),
                dtype,
                message: "The y column of an area chart must be Numeric.".to_string(),
            });
        }

        let (y_vals, _y_w) = to_f64(df, y_col)?;

        // ------------------------------------------------------------------
        // 4. Optional color_by column (Categorical only)
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
                        message: "color_by for an area chart must be a Categorical column \
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
        //    Each series: (label, color, Vec<(x_f64, Option<y_f64>)>)
        // ------------------------------------------------------------------
        let theme_cfg = ThemeConfig::from(&config.theme);

        let series_list: Vec<(String, String, Vec<(f64, Option<f64>)>)> =
            if color_vals.is_none() {
                let color = theme_cfg.palette[0].to_string();
                let pts: Vec<(f64, Option<f64>)> = (0..n_rows)
                    .filter_map(|i| x_f64[i].map(|xv| (xv, y_vals[i])))
                    .collect();
                vec![("".to_string(), color, pts)]
            } else {
                let cv = color_vals.as_ref().unwrap();
                // Collect unique category labels preserving first-seen order.
                let mut seen: Vec<String> = Vec::new();
                for opt in cv {
                    let label = opt.clone().unwrap_or_else(|| "null".to_string());
                    if !seen.contains(&label) {
                        seen.push(label);
                    }
                }

                seen.iter().enumerate().map(|(idx, label)| {
                    let color = if label == "null" {
                        NULL_COLOR.to_string()
                    } else {
                        theme_cfg.palette[idx % theme_cfg.palette.len()].to_string()
                    };
                    let pts: Vec<(f64, Option<f64>)> = (0..n_rows)
                        .filter(|&i| {
                            cv[i].as_deref().unwrap_or("null") == label.as_str()
                        })
                        .filter_map(|i| x_f64[i].map(|xv| (xv, y_vals[i])))
                        .collect();
                    (label.clone(), color, pts)
                }).collect()
            };

        // ------------------------------------------------------------------
        // 6. FillMode::Between validation (must have exactly 2 series)
        // ------------------------------------------------------------------
        if config.fill_mode == FillMode::Between {
            let n = series_list.len();
            if n != 2 {
                return Err(CharcoalError::InsufficientData {
                    col:      config.color_by.clone().unwrap_or_else(|| y_col.to_string()),
                    required: 2,
                    got:      n,
                });
            }
        }

        // ------------------------------------------------------------------
        // 7. Null-y warnings (emitted regardless of NullPolicy)
        // ------------------------------------------------------------------
        for (label, _, pts) in &series_list {
            let y_null_count = pts.iter().filter(|(_, y)| y.is_none()).count();
            if y_null_count > 0 {
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
        // 8. Axis ranges
        // ------------------------------------------------------------------
        let all_x: Vec<f64> = series_list
            .iter()
            .flat_map(|(_, _, pts)| pts.iter().map(|(x, _)| *x))
            .collect();

        let all_y_raw: Vec<f64> = series_list
            .iter()
            .flat_map(|(_, _, pts)| pts.iter().filter_map(|(_, y)| *y))
            .collect();

        let x_min      = all_x.iter().cloned().fold(f64::INFINITY,     f64::min);
        let x_max      = all_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min_data = all_y_raw.iter().cloned().fold(f64::INFINITY,     f64::min);
        let y_max_data = all_y_raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let (x_min, x_max) = if x_min.is_infinite() { (0.0, 1.0) } else { (x_min, x_max) };
        let (y_min_data, y_max_data) =
            if y_min_data.is_infinite() { (0.0, 1.0) } else { (y_min_data, y_max_data) };

        // The y-axis must always include the baseline.
        let baseline_y = match config.fill_mode {
            FillMode::ToZero    => 0.0_f64,
            FillMode::ToMinimum => y_min_data,
            FillMode::Between   => y_min_data,
        };

        // When stacking, compute an upper bound = sum of per-series maxima.
        let y_max_for_range = if config.stacked && series_list.len() > 1 {
            series_list
                .iter()
                .map(|(_, _, pts)| {
                    pts.iter().filter_map(|(_, y)| *y).fold(0.0_f64, f64::max)
                })
                .sum()
        } else {
            y_max_data
        };

        let y_range_min = baseline_y.min(y_min_data);
        let y_range_max = y_max_for_range;

        let x_tick_vals = nice_ticks(x_min, x_max, 6);
        let y_tick_vals = nice_ticks(y_range_min, y_range_max, 6);

        // ------------------------------------------------------------------
        // 9. Canvas and scales
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
        // 10. Render polygons
        // ------------------------------------------------------------------
        let stroke_w = config.stroke_width;
        let dash     = config.dash_style.stroke_dasharray().unwrap_or("");

        let mut elements:      Vec<String>            = Vec::new();
        let mut legend_entries: Vec<(String, String)> = Vec::new();

        // Apply null policy to each series: produces pixel-space segments.
        let resolved_series: Vec<(String, String, Vec<Vec<(f64, f64)>>)> = series_list
            .iter()
            .map(|(label, color, pts)| {
                let segs = match config.null_policy {
                    NullPolicy::Skip        => segments_skip(pts, &x_scale, &y_scale),
                    NullPolicy::Interpolate => {
                        let filled = interpolate_nulls(pts);
                        segments_skip(&filled, &x_scale, &y_scale)
                    }
                };
                (label.clone(), color.clone(), segs)
            })
            .collect();

        // ------------------------------------------------------------------
        // 10a. FillMode::Between — polygon between exactly two series.
        // ------------------------------------------------------------------
        if config.fill_mode == FillMode::Between {
            let (label0, color0, segs0) = &resolved_series[0];
            let (label1, color1, segs1) = &resolved_series[1];

            for seg0 in segs0 {
                if seg0.len() < 2 { continue; }
                let x0_start = seg0.first().unwrap().0;
                let x0_end   = seg0.last().unwrap().0;

                // Collect all series-1 points that overlap this x range.
                let overlap: Vec<(f64, f64)> = segs1
                    .iter()
                    .flat_map(|s| s.iter().copied())
                    .filter(|(px, _)| *px >= x0_start - 0.5 && *px <= x0_end + 0.5)
                    .collect();
                if overlap.len() < 2 { continue; }

                // Polygon: forward along series 0, backward along series 1.
                let mut poly_pts: Vec<(f64, f64)> = seg0.clone();
                let mut rev = overlap.clone();
                rev.reverse();
                poly_pts.extend(rev);
                elements.push(geometry::polygon(&poly_pts, color0, color0, FILL_OPACITY));

                // Stroke each edge.
                elements.push(geometry::polyline(seg0, color0, stroke_w, dash));
                elements.push(geometry::polyline(&overlap, color1, stroke_w, dash));
            }

            if config.color_by.is_some() {
                legend_entries.push((label0.clone(), color0.clone()));
                legend_entries.push((label1.clone(), color1.clone()));
            }
        }
        // ------------------------------------------------------------------
        // 10b. Non-stacked ToZero / ToMinimum
        // ------------------------------------------------------------------
        else if !config.stacked || series_list.len() <= 1 {
            let baseline_px = y_scale.map(baseline_y);

            for (label, color, segs) in &resolved_series {
                for seg in segs {
                    if seg.len() < 2 { continue; }
                    let poly = closed_polygon_with_baseline(seg, baseline_px);
                    elements.push(geometry::polygon(&poly, color, color, FILL_OPACITY));
                    // Crisp top-edge stroke.
                    elements.push(geometry::polyline(seg, color, stroke_w, dash));
                }
                if config.color_by.is_some() {
                    legend_entries.push((label.clone(), color.clone()));
                }
            }
        }
        // ------------------------------------------------------------------
        // 10c. Stacked area
        // ------------------------------------------------------------------
        else {
            // We accumulate per-x cumulative sums in data-space, then scale
            // both the top and baseline of each layer to pixel coordinates.
            let mut cumulative_y: std::collections::HashMap<u64, f64> =
                std::collections::HashMap::new();

            for (series_idx, (label, color, _)) in resolved_series.iter().enumerate() {
                let (_, _, raw_pts) = &series_list[series_idx];

                // Apply null policy in data space.
                let effective_pts: Vec<(f64, Option<f64>)> = match config.null_policy {
                    NullPolicy::Skip        => raw_pts.clone(),
                    NullPolicy::Interpolate => interpolate_nulls(raw_pts),
                };

                // Stacked top = data_y + cumulative[x].
                let stacked_pts: Vec<(f64, Option<f64>)> = effective_pts
                    .iter()
                    .map(|&(x, y_opt)| {
                        let key = x.to_bits();
                        let cum = *cumulative_y.get(&key).unwrap_or(&0.0);
                        (x, y_opt.map(|y| y + cum))
                    })
                    .collect();

                // Baseline for this layer = cumulative top of all previous layers.
                let baseline_pts: Vec<(f64, Option<f64>)> = effective_pts
                    .iter()
                    .map(|&(x, _)| {
                        let key = x.to_bits();
                        let cum = *cumulative_y.get(&key).unwrap_or(&0.0);
                        (x, Some(cum))
                    })
                    .collect();

                // Update cumulative sums for the next series.
                for &(x, y_opt) in &effective_pts {
                    if let Some(y) = y_opt {
                        let key = x.to_bits();
                        *cumulative_y.entry(key).or_insert(0.0) += y;
                    }
                }

                // Scale stacked and baseline pts to pixel space, split at nulls.
                let top_segs    = segments_skip(&stacked_pts,  &x_scale, &y_scale);
                let bottom_segs = segments_skip(&baseline_pts, &x_scale, &y_scale);

                // Pair top and bottom segments positionally.
                let n_segs = top_segs.len().min(bottom_segs.len());
                for seg_idx in 0..n_segs {
                    let top = &top_segs[seg_idx];
                    let bot = &bottom_segs[seg_idx];
                    if top.len() < 2 { continue; }

                    // Polygon: forward along top, backward along bottom.
                    let mut poly: Vec<(f64, f64)> = top.clone();
                    let mut rev_bot = bot.clone();
                    rev_bot.reverse();
                    poly.extend(rev_bot);
                    elements.push(geometry::polygon(&poly, color, color, FILL_OPACITY));
                    elements.push(geometry::polyline(top, color, stroke_w, dash));
                }

                if config.color_by.is_some() {
                    legend_entries.push((label.clone(), color.clone()));
                }
            }
        }

        // ------------------------------------------------------------------
        // 11. Axes
        // ------------------------------------------------------------------
        let x_labels = if x_is_temporal {
            let range_ms  = (x_max - x_min) as i64;
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
        // 12. Legend (only when color_by is set)
        // ------------------------------------------------------------------
        let legend: Option<Vec<(String, String)>> = if legend_entries.is_empty() {
            None
        } else {
            Some(legend_entries)
        };

        // ------------------------------------------------------------------
        // 13. Assemble
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
// Rendering helpers
// ---------------------------------------------------------------------------

/// Builds a closed polygon by tracing `top_pts` forward then returning along
/// the horizontal baseline at pixel-y `baseline_px`.
///
/// The resulting slice is passed to `geometry::polygon()`.
fn closed_polygon_with_baseline(
    top_pts:     &[(f64, f64)],
    baseline_px: f64,
) -> Vec<(f64, f64)> {
    debug_assert!(!top_pts.is_empty());

    let mut pts = top_pts.to_vec();
    // Bottom-right then bottom-left at the baseline.
    pts.push((top_pts.last().unwrap().0,  baseline_px));
    pts.push((top_pts.first().unwrap().0, baseline_px));
    pts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn simple_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("x", &[1.0f64, 2.0, 3.0, 4.0, 5.0]),
            Series::new("y", &[2.0f64, 4.0, 3.0, 5.0, 1.0]),
        ]).unwrap()
    }

    fn two_series_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("x",   &[1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0]),
            Series::new("y",   &[2.0f64, 4.0, 3.0, 1.0, 2.0, 1.5]),
            Series::new("grp", &["A", "A", "A", "B", "B", "B"]),
        ]).unwrap()
    }

    fn three_series_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("x",   &[1.0f64, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            Series::new("y",   &[2.0f64, 4.0, 3.0, 1.0, 2.0, 1.5, 0.5, 1.0, 0.8]),
            Series::new("grp", &["A", "A", "A", "B", "B", "B", "C", "C", "C"]),
        ]).unwrap()
    }

    fn null_y_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("x", &[1.0f64, 2.0, 3.0, 4.0, 5.0]),
            Series::new("y", &[Some(2.0f64), Some(4.0), None, Some(5.0), Some(1.0)]),
        ]).unwrap()
    }

    // ── Basic build ───────────────────────────────────────────────────────────

    #[test]
    fn basic_area_chart_builds() {
        let df = simple_df();
        assert!(AreaBuilder::new(&df).x("x").y("y").build().is_ok());
    }

    #[test]
    fn output_is_svg() {
        let df  = simple_df();
        let svg = AreaBuilder::new(&df).x("x").y("y").build().unwrap().svg().to_string();
        assert!(svg.contains("<svg"), "output must be SVG");
    }

    #[test]
    fn output_dimensions_match_constants() {
        let df    = simple_df();
        let chart = AreaBuilder::new(&df).x("x").y("y").build().unwrap();
        assert_eq!(chart.width(),  CANVAS_WIDTH);
        assert_eq!(chart.height(), CANVAS_HEIGHT);
    }

    #[test]
    fn clean_data_emits_no_warnings() {
        let df = simple_df();
        assert!(AreaBuilder::new(&df).x("x").y("y").build().unwrap().warnings().is_empty());
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn missing_x_column_returns_column_not_found() {
        let df  = simple_df();
        let err = AreaBuilder::new(&df).x("no_such").y("y").build().unwrap_err();
        assert!(matches!(err, CharcoalError::ColumnNotFound { .. }));
    }

    #[test]
    fn missing_y_column_returns_column_not_found() {
        let df  = simple_df();
        let err = AreaBuilder::new(&df).x("x").y("no_such").build().unwrap_err();
        assert!(matches!(err, CharcoalError::ColumnNotFound { .. }));
    }

    #[test]
    fn categorical_y_returns_unsupported_column() {
        let df = DataFrame::new(vec![
            Series::new("x", &[1.0f64, 2.0]),
            Series::new("y", &["a", "b"]),
        ]).unwrap();
        let err = AreaBuilder::new(&df).x("x").y("y").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { .. }));
    }

    #[test]
    fn row_limit_exceeded_returns_data_too_large() {
        let df  = simple_df(); // 5 rows
        let err = AreaBuilder::new(&df).x("x").y("y").row_limit(3).build().unwrap_err();
        match err {
            CharcoalError::DataTooLarge { rows, limit, .. } => {
                assert_eq!(rows, 5);
                assert_eq!(limit, 3);
            }
            other => panic!("expected DataTooLarge; got {other:?}"),
        }
    }

    // ── FillMode::ToZero — closed polygon present ─────────────────────────────

    #[test]
    fn fill_mode_to_zero_produces_closed_polygon() {
        let df  = simple_df();
        let svg = AreaBuilder::new(&df)
            .x("x").y("y")
            .fill_mode(FillMode::ToZero)
            .build().unwrap().svg().to_string();
        // geometry::polygon closes the path with an SVG "Z" (or "z") command.
        assert!(
            svg.contains('Z') || svg.contains('z'),
            "ToZero area chart must contain a closed polygon path (Z command)"
        );
    }

    #[test]
    fn closed_polygon_helper_baseline_coords() {
        // Test closed_polygon_with_baseline directly.
        let top      = vec![(10.0, 50.0), (20.0, 40.0), (30.0, 60.0)];
        let baseline = 200.0_f64;
        let poly     = closed_polygon_with_baseline(&top, baseline);

        // 3 top points + 2 baseline corners = 5 total.
        assert_eq!(poly.len(), 5, "must have 3 top + 2 baseline points");
        // Both baseline points are at baseline_px.
        assert!((poly[3].1 - baseline).abs() < 1e-9, "bottom-right y must equal baseline");
        assert!((poly[4].1 - baseline).abs() < 1e-9, "bottom-left y must equal baseline");
        // bottom-right x == last top x; bottom-left x == first top x.
        assert!((poly[3].0 - 30.0).abs() < 1e-9, "bottom-right x must match last top point");
        assert!((poly[4].0 - 10.0).abs() < 1e-9, "bottom-left x must match first top point");
    }

    // ── FillMode::ToMinimum ───────────────────────────────────────────────────

    #[test]
    fn fill_mode_to_minimum_builds_successfully() {
        let df = simple_df();
        assert!(
            AreaBuilder::new(&df)
                .x("x").y("y")
                .fill_mode(FillMode::ToMinimum)
                .build()
                .is_ok()
        );
    }

    // ── FillMode::Between ─────────────────────────────────────────────────────

    #[test]
    fn fill_mode_between_with_two_series_builds() {
        let df = two_series_df();
        let r  = AreaBuilder::new(&df)
            .x("x").y("y").color_by("grp")
            .fill_mode(FillMode::Between)
            .build();
        assert!(r.is_ok(), "Between with 2 series must build; got {r:?}");
    }

    #[test]
    fn fill_mode_between_with_one_series_returns_insufficient_data() {
        let df  = simple_df();
        let err = AreaBuilder::new(&df)
            .x("x").y("y")
            .fill_mode(FillMode::Between)
            .build()
            .unwrap_err();
        assert!(
            matches!(err, CharcoalError::InsufficientData { required: 2, got: 1, .. }),
            "Between with 1 series must return InsufficientData(required=2, got=1); got {err:?}"
        );
    }

    #[test]
    fn fill_mode_between_with_three_series_returns_insufficient_data() {
        let df  = three_series_df();
        let err = AreaBuilder::new(&df)
            .x("x").y("y").color_by("grp")
            .fill_mode(FillMode::Between)
            .build()
            .unwrap_err();
        assert!(
            matches!(err, CharcoalError::InsufficientData { required: 2, got: 3, .. }),
            "Between with 3 series must return InsufficientData(required=2, got=3); got {err:?}"
        );
    }

    // ── Stacking ──────────────────────────────────────────────────────────────

    #[test]
    fn stacked_area_builds() {
        let df = two_series_df();
        assert!(
            AreaBuilder::new(&df)
                .x("x").y("y").color_by("grp")
                .stacked(true)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn stacked_area_produces_multiple_path_elements() {
        // Stacking two series should emit at least 2 polygon paths (one per series).
        let df    = two_series_df();
        let chart = AreaBuilder::new(&df)
            .x("x").y("y").color_by("grp")
            .stacked(true)
            .build()
            .unwrap();
        let path_count = chart.svg().matches("<path").count();
        assert!(
            path_count >= 2,
            "stacked chart must produce ≥ 2 path elements; found {path_count}"
        );
    }

    #[test]
    fn stacked_without_color_by_falls_back_to_single_series() {
        let df = simple_df();
        assert!(AreaBuilder::new(&df).x("x").y("y").stacked(true).build().is_ok());
    }

    // ── Null handling ─────────────────────────────────────────────────────────

    #[test]
    fn null_y_emits_warning() {
        let df    = null_y_df();
        let chart = AreaBuilder::new(&df).x("x").y("y").build().unwrap();
        let warned = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, count: 1, .. } if col.starts_with('y')
        ));
        assert!(warned, "null y must emit NullsSkipped warning");
    }

    #[test]
    fn null_policy_skip_breaks_polygon_at_null() {
        // Interior null at index 2 splits the series into [0,1] and [3,4].
        // Each produces its own closed polygon → at least 2 Z commands.
        let df    = null_y_df();
        let chart = AreaBuilder::new(&df)
            .x("x").y("y")
            .null_policy(NullPolicy::Skip)
            .build()
            .unwrap();
        let z_count = chart.svg().chars().filter(|&c| c == 'Z').count();
        assert!(
            z_count >= 2,
            "Skip with interior null must produce ≥ 2 closed paths; found {z_count} Z command(s)"
        );
    }

    #[test]
    fn null_policy_interpolate_still_emits_warning() {
        // Interpolate fills the gap visually but must still warn about the null.
        let df    = null_y_df();
        let chart = AreaBuilder::new(&df)
            .x("x").y("y")
            .null_policy(NullPolicy::Interpolate)
            .build()
            .unwrap();
        let warned = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { .. }
        ));
        assert!(warned, "Interpolate must still emit NullsSkipped warning");
    }

    #[test]
    fn null_policy_interpolate_produces_closed_polygon() {
        let df  = null_y_df();
        let svg = AreaBuilder::new(&df)
            .x("x").y("y")
            .null_policy(NullPolicy::Interpolate)
            .build()
            .unwrap()
            .svg()
            .to_string();
        assert!(svg.contains('Z'), "Interpolate must produce at least one closed polygon");
    }

    // ── Optional setters ─────────────────────────────────────────────────────

    #[test]
    fn color_by_produces_legend_with_series_labels() {
        let df  = two_series_df();
        let svg = AreaBuilder::new(&df)
            .x("x").y("y").color_by("grp")
            .build().unwrap().svg().to_string();
        assert!(svg.contains("A"), "legend must contain 'A'");
        assert!(svg.contains("B"), "legend must contain 'B'");
    }

    #[test]
    fn title_appears_in_svg() {
        let df  = simple_df();
        let svg = AreaBuilder::new(&df)
            .x("x").y("y").title("My Area Chart")
            .build().unwrap().svg().to_string();
        assert!(svg.contains("My Area Chart"));
    }

    #[test]
    fn stroke_width_clamped_to_minimum() {
        let df = simple_df();
        assert!(AreaBuilder::new(&df).stroke_width(0.0).config.stroke_width >= 0.5);
    }

    #[test]
    fn defaults_are_sane() {
        let df = simple_df();
        let b  = AreaBuilder::new(&df);
        assert_eq!(b.config.null_policy, NullPolicy::Skip);
        assert_eq!(b.config.fill_mode,   FillMode::ToZero);
        assert!(!b.config.stacked);
        assert_eq!(b.config.row_limit,   DEFAULT_ROW_LIMIT);
    }

    #[test]
    fn single_row_does_not_panic() {
        let df = DataFrame::new(vec![
            Series::new("x", &[1.0f64]),
            Series::new("y", &[5.0f64]),
        ]).unwrap();
        assert!(AreaBuilder::new(&df).x("x").y("y").build().is_ok());
    }
}