//! Bar chart builder (`Chart::bar(&df)`).
//!
//! Follows the same typestate pattern as `scatter.rs` and `line.rs`.
//!
//! # Column roles
//!
//! | Column | Required dtype | Role |
//! |--------|---------------|------|
//! | `x`    | Categorical   | Group labels on the categorical axis |
//! | `y`    | Numeric       | Bar height (vertical) or bar width (horizontal) |
//!
//! Passing a Numeric column as `x` returns [`CharcoalError::UnsupportedColumn`] with a
//! descriptive message — this is the most common caller mistake.
//!
//! # Orientation
//!
//! [`Orientation::Vertical`] (default): categories on the x-axis, values on y.
//! [`Orientation::Horizontal`]: categories on the y-axis, values on x.
//! The axis computation logic is identical — orientation only controls which
//! pixel dimension each scale is mapped to.
//!
//! # Stacking
//!
//! When `.stacked(true)` and `color_by` is set, bars for the same x category are
//! stacked rather than grouped side-by-side. Cumulative y-values are computed per
//! category; each stack segment covers `[previous_top, current_top]`.
//!
//! # Null handling
//!
//! Null x values are treated as the category label `"null"`.  
//! Null y values are omitted — the bar for that category is absent, leaving a visible gap.
//! [`CharcoalWarning::NullsSkipped`] is emitted for every skipped y null.

use polars::frame::DataFrame;

use crate::charts::{Chart, Orientation};
// NOTE: `Orientation` must be added to `src/chart/mod.rs` alongside `Chart`,
// `NullPolicy`, and `DashStyle`. See the addition required at the end of this file.
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

const CANVAS_WIDTH:      u32   = 800;
const CANVAS_HEIGHT:     u32   = 500;
const DEFAULT_ROW_LIMIT: usize = 1_000_000;
/// Neutral grey for the synthetic null-category. Matches scatter.rs / line.rs.
const NULL_COLOR: &str = "#AAAAAA";
/// Fractional gap between adjacent grouped bars within a category band (0 = no gap).
const BAR_GROUP_GAP: f64 = 0.1;
/// Fractional padding between category bands (fraction of band width on each side).
const BAND_PADDING: f64 = 0.15;

// ---------------------------------------------------------------------------
// Accumulated configuration
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct BarConfig {
    pub x_col:      Option<String>,
    pub y_col:      Option<String>,
    pub color_by:   Option<String>,
    pub title:      Option<String>,
    pub x_label:    Option<String>,
    pub y_label:    Option<String>,
    pub theme:      Theme,
    pub orientation: Orientation,
    pub stacked:    bool,
    pub row_limit:  usize,
}

impl Default for BarConfig {
    fn default() -> Self {
        Self {
            x_col:       None,
            y_col:       None,
            color_by:    None,
            title:       None,
            x_label:     None,
            y_label:     None,
            theme:       Theme::Default,
            orientation: Orientation::Vertical,
            stacked:     false,
            row_limit:   DEFAULT_ROW_LIMIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder states
// ---------------------------------------------------------------------------

/// Initial bar builder — no required fields set yet.
pub struct BarBuilder<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: BarConfig,
}

/// x-axis column set; y still required.
pub struct BarWithX<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: BarConfig,
}

/// Both required fields set — `.build()` is available.
pub struct BarWithXY<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: BarConfig,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<'df> BarBuilder<'df> {
    pub(crate) fn new(df: &'df DataFrame) -> Self {
        Self { df, config: BarConfig::default() }
    }
}

// ---------------------------------------------------------------------------
// Optional setter methods — stamped onto all three states via macro so
// the compiler sees each as returning `Self`, preserving the typestate.
// ---------------------------------------------------------------------------

macro_rules! impl_bar_optional_setters {
    ($t:ty) => {
        impl<'df> $t {
            /// Column used to split bars into colour-coded groups or stacks.
            ///
            /// Must be Categorical. Null values are collected into a synthetic
            /// `"null"` group and coloured grey.
            pub fn color_by(mut self, col: &str) -> Self {
                self.config.color_by = Some(col.to_string());
                self
            }

            /// Chart title rendered above the plot area.
            pub fn title(mut self, title: &str) -> Self {
                self.config.title = Some(title.to_string());
                self
            }

            /// Label rendered below the x-axis (or beside the y-axis for horizontal charts).
            pub fn x_label(mut self, label: &str) -> Self {
                self.config.x_label = Some(label.to_string());
                self
            }

            /// Label rendered beside the y-axis (or below the x-axis for horizontal charts).
            pub fn y_label(mut self, label: &str) -> Self {
                self.config.y_label = Some(label.to_string());
                self
            }

            /// Visual theme. Defaults to [`Theme::Default`].
            pub fn theme(mut self, theme: Theme) -> Self {
                self.config.theme = theme;
                self
            }

            /// Bar orientation. Defaults to [`Orientation::Vertical`].
            pub fn orientation(mut self, orientation: Orientation) -> Self {
                self.config.orientation = orientation;
                self
            }

            /// When `true` and `color_by` is set, stack bars for the same category
            /// rather than grouping them side-by-side. Defaults to `false`.
            pub fn stacked(mut self, stacked: bool) -> Self {
                self.config.stacked = stacked;
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

impl_bar_optional_setters!(BarBuilder<'df>);
impl_bar_optional_setters!(BarWithX<'df>);
impl_bar_optional_setters!(BarWithXY<'df>);

// ---------------------------------------------------------------------------
// Required field transitions
// ---------------------------------------------------------------------------

impl<'df> BarBuilder<'df> {
    /// Set the categorical x-axis column (group labels).
    ///
    /// Column validation is deferred to `.build()` — no `?` needed when chaining.
    pub fn x(mut self, col: &str) -> BarWithX<'df> {
        self.config.x_col = Some(col.to_string());
        BarWithX { df: self.df, config: self.config }
    }
}

impl<'df> BarWithX<'df> {
    /// Set the numeric y-axis column (bar heights). Unlocks `.build()`.
    ///
    /// Column validation is deferred to `.build()` — no `?` needed when chaining.
    pub fn y(mut self, col: &str) -> BarWithXY<'df> {
        self.config.y_col = Some(col.to_string());
        BarWithXY { df: self.df, config: self.config }
    }
}

// ---------------------------------------------------------------------------
// .build()
// ---------------------------------------------------------------------------

impl<'df> BarWithXY<'df> {
    /// Validate columns, aggregate by category, render bars, and return a [`Chart`].
    ///
    /// # Errors
    ///
    /// - [`CharcoalError::DataTooLarge`] if `df.height() > row_limit`.
    /// - [`CharcoalError::ColumnNotFound`] for any missing column name.
    /// - [`CharcoalError::UnsupportedColumn`] if x is Numeric/Temporal/Unsupported, y is not
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
        // 2. Validate and normalize x (must be Categorical)
        // ------------------------------------------------------------------
        let x_col = config.x_col.as_deref().unwrap(); // always present (typestate)
        let x_viz = classify_column(df, x_col, None)?;

        if x_viz == VizDtype::Numeric || x_viz == VizDtype::Temporal {
            let dtype = df.schema().get(x_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     x_col.to_string(),
                dtype,
                message: format!(
                    "The x column of a bar chart must be Categorical (String or Boolean), \
                     not {:?}. Did you mean to use it as the y (value) column instead? \
                     Bar charts group by a categorical x column and measure a numeric y column.",
                    df.schema().get(x_col).unwrap()
                ),
            });
        }
        if x_viz == VizDtype::Unsupported {
            let dtype = df.schema().get(x_col).unwrap().clone();
            return Err(CharcoalError::UnsupportedColumn {
                col:     x_col.to_string(),
                dtype,
                message: "The x column of a bar chart must be Categorical (String or Boolean)."
                    .to_string(),
            });
        }

        // Normalize x: nulls become the string "null"
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
                message: "The y column of a bar chart must be Numeric.".to_string(),
            });
        }

        let (y_vals, _y_w) = to_f64(df, y_col)?;
        // _y_w discarded — null y rows are handled below with richer context.

        // ------------------------------------------------------------------
        // 4. Optional color_by (Categorical only)
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
                        message: "color_by for a bar chart must be a Categorical column \
                                  (String or Boolean dtype).".to_string(),
                    });
                }
                let (vals, w) = to_string(df, col)?;
                warnings.extend(w);
                Some(vals)
            }
        };

        // ------------------------------------------------------------------
        // 5. Build per-series data: (series_label, color, Map<category, y_value>)
        //
        // A "series" is one distinct color_by value (or the single default series
        // when color_by is not set). For each series, aggregate rows by x category:
        //   - sum y values within the same (category, series) cell
        //   - skip rows where y is None (emit NullsSkipped warning)
        // ------------------------------------------------------------------
        let theme_cfg = ThemeConfig::from(&config.theme);

        // Collect ordered list of x categories (first-seen order, null last)
        let mut x_categories: Vec<String> = Vec::new();
        for v in &x_vals {
            if !x_categories.contains(v) {
                x_categories.push(v.clone());
            }
        }

        // Collect ordered list of series labels
        let series_labels: Vec<(String, String)> = if let Some(cv) = &color_vals {
            let mut order: Vec<Option<String>> = Vec::new();
            for v in cv {
                if !order.contains(v) {
                    order.push(v.clone());
                }
            }
            let mut palette_idx = 0usize;
            order
                .into_iter()
                .map(|opt| {
                    let label = opt.as_deref().unwrap_or("null").to_string();
                    let color = if opt.is_none() {
                        NULL_COLOR.to_string()
                    } else {
                        let c = theme_cfg.palette[palette_idx % theme_cfg.palette.len()]
                            .to_string();
                        palette_idx += 1;
                        c
                    };
                    (label, color)
                })
                .collect()
        } else {
            vec![("".to_string(), theme_cfg.palette[0].to_string())]
        };

        // For each (series, category), sum all non-null y values.
        // `data[series_idx][cat_idx]` = Option<f64>
        // None means no non-null rows contributed (bar is absent / gap).
        let n_series = series_labels.len();
        let n_cats   = x_categories.len();
        let mut data: Vec<Vec<Option<f64>>> = vec![vec![None; n_cats]; n_series];
        let mut null_y_counts: Vec<usize> = vec![0; n_series];

        for row in 0..n_rows {
            let cat_idx = x_categories
                .iter()
                .position(|c| c == &x_vals[row])
                .unwrap(); // always found — we built x_categories from x_vals

            let series_idx = if let Some(cv) = &color_vals {
                let label = cv[row].as_deref().unwrap_or("null");
                series_labels
                    .iter()
                    .position(|(l, _)| l == label)
                    .unwrap()
            } else {
                0
            };

            match y_vals[row] {
                None => {
                    null_y_counts[series_idx] += 1;
                }
                Some(y) => {
                    let cell = &mut data[series_idx][cat_idx];
                    *cell = Some(cell.unwrap_or(0.0) + y);
                }
            }
        }

        // Emit NullsSkipped warnings for null y values
        for (idx, &count) in null_y_counts.iter().enumerate() {
            if count > 0 {
                let col_label = if series_labels[idx].0.is_empty() {
                    y_col.to_string()
                } else {
                    format!("{} ({})", y_col, series_labels[idx].0)
                };
                warnings.push(CharcoalWarning::NullsSkipped {
                    col:   col_label,
                    count,
                });
            }
        }

        // ------------------------------------------------------------------
        // 6. Compute y axis range
        //
        // For stacked mode: the max is the maximum cumulative sum across all
        //   categories. For grouped mode: the max is the maximum individual value.
        // ------------------------------------------------------------------
        let y_axis_max: f64 = if config.stacked && n_series > 1 {
            // For each category, sum all series values; take the max of these sums.
            (0..n_cats)
                .map(|ci| {
                    data.iter()
                        .filter_map(|series| series[ci])
                        .fold(0.0_f64, |acc, v| acc + v)
                })
                .fold(0.0_f64, f64::max)
        } else {
            data.iter()
                .flat_map(|s| s.iter())
                .filter_map(|v| *v)
                .fold(0.0_f64, f64::max)
        };

        // Always start the value axis at 0 for bars.
        let y_axis_min = 0.0_f64;

        // Guard against all-null / empty data.
        let y_axis_max = if y_axis_max == 0.0 { 1.0 } else { y_axis_max };

        let y_tick_vals = nice_ticks(y_axis_min, y_axis_max, 6);

        // ------------------------------------------------------------------
        // 7. Canvas and scales
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

        // The categorical scale maps category labels to pixel centers.
        // The value scale maps numeric y values to the perpendicular pixel axis.
        let cat_positions = categorical_scale(&x_categories, 0.0, 1.0); // normalized 0..1

        // value scale: data 0 → pixel bottom/left, data max → pixel top/right
        let val_scale = LinearScale::new(
            *y_tick_vals.first().unwrap(),
            *y_tick_vals.last().unwrap(),
            if config.orientation == Orientation::Vertical { oy + ph } else { ox },
            if config.orientation == Orientation::Vertical { oy       } else { ox + pw },
        );

        // ------------------------------------------------------------------
        // 8. Render bar rectangles
        //
        // Each category band occupies a fixed pixel slice of the categorical axis.
        // Within the band, bars are either grouped side-by-side or stacked.
        // ------------------------------------------------------------------
        let band_pixel_size = match config.orientation {
            Orientation::Vertical   => pw / n_cats as f64,
            Orientation::Horizontal => ph / n_cats as f64,
        };
        let band_pad = band_pixel_size * BAND_PADDING;

        let mut elements: Vec<String> = Vec::new();

        let zero_px = val_scale.map(0.0);

        for (ci, (_cat_label, cat_norm)) in cat_positions.iter().enumerate() {
            // Convert the normalized [0,1] center to actual pixel position.
            let band_center_px = match config.orientation {
                Orientation::Vertical   => ox + cat_norm * pw,
                Orientation::Horizontal => oy + cat_norm * ph,
            };
            let bar_area_start = band_center_px - band_pixel_size / 2.0 + band_pad;
            let bar_area_width = band_pixel_size - 2.0 * band_pad;

            if config.stacked && n_series > 1 {
                // ── Stacked ──────────────────────────────────────────────
                // Accumulate in data space so the pixel mapping stays correct.
                let mut cum_data: f64 = 0.0;
                for (si, (_label, color)) in series_labels.iter().enumerate() {
                    let y_val = match data[si][ci] {
                        None    => continue,
                        Some(v) => v,
                    };
                    let base_px = val_scale.map(cum_data);
                    let top_px  = val_scale.map(cum_data + y_val);
                    cum_data += y_val;

                    elements.push(bar_rect(
                        &config.orientation,
                        bar_area_start,
                        bar_area_width,
                        base_px,
                        top_px,
                        color,
                    ));
                }
            } else {
                // ── Grouped ──────────────────────────────────────────────
                let visible_series: Vec<(usize, &str)> = series_labels
                    .iter()
                    .enumerate()
                    .filter(|(si, _)| data[*si][ci].is_some())
                    .map(|(si, (_, color))| (si, color.as_str()))
                    .collect();

                let n_visible = visible_series.len().max(1);
                let bar_gap   = bar_area_width * BAR_GROUP_GAP / n_visible as f64;
                let bar_w     = (bar_area_width - bar_gap * (n_visible as f64 - 1.0))
                    / n_visible as f64;

                for (group_i, (si, color)) in visible_series.iter().enumerate() {
                    let y_val = match data[*si][ci] {
                        None    => continue,
                        Some(v) => v,
                    };
                    let bar_start = bar_area_start
                        + group_i as f64 * (bar_w + bar_gap);
                    let top_px  = val_scale.map(y_val);

                    elements.push(bar_rect(
                        &config.orientation,
                        bar_start,
                        bar_w,
                        zero_px,
                        top_px,
                        color,
                    ));
                }
            }
        }

        // ------------------------------------------------------------------
        // 9. Axis SVG
        //
        // Categorical axis: tick marks centered on each category band.
        // Value axis:       numeric tick marks.
        // ------------------------------------------------------------------
        let y_labels  = tick_labels_numeric(&y_tick_vals);
        let val_ticks = build_tick_marks(&y_tick_vals, &y_labels, &val_scale);

        // Build categorical tick marks manually (pixel positions from band centers).
        let cat_tick_marks: Vec<TickMark> = cat_positions
            .iter()
            .map(|(cat_label, norm)| {
                let px = match config.orientation {
                    Orientation::Vertical   => ox + norm * pw,
                    Orientation::Horizontal => oy + norm * ph,
                };
                TickMark {
                    data_value: *norm,
                    pixel_pos:  px,
                    label:      cat_label.clone(),
                }
            })
            .collect();

        // Fake scale for the categorical axis (pixel_min..pixel_max are the axis ends).
        let cat_scale = LinearScale::new(0.0, 1.0, 0.0, 1.0); // not used for mapping — ticks carry px

        let (x_axis, y_axis) = match config.orientation {
            Orientation::Vertical => {
                // Categorical on horizontal, values on vertical
                let x_axis = compute_axis(
                    &cat_scale, &cat_tick_marks, AxisOrientation::Horizontal,
                    ox, oy, pw, ph, &theme_cfg,
                );
                let y_axis = compute_axis(
                    &val_scale, &val_ticks, AxisOrientation::Vertical,
                    ox, oy, pw, ph, &theme_cfg,
                );
                (x_axis, y_axis)
            }
            Orientation::Horizontal => {
                // Values on horizontal, categorical on vertical
                let x_axis = compute_axis(
                    &val_scale, &val_ticks, AxisOrientation::Horizontal,
                    ox, oy, pw, ph, &theme_cfg,
                );
                let y_axis = compute_axis(
                    &cat_scale, &cat_tick_marks, AxisOrientation::Vertical,
                    ox, oy, pw, ph, &theme_cfg,
                );
                (x_axis, y_axis)
            }
        };

        // ------------------------------------------------------------------
        // 10. Legend (only when color_by is set)
        // ------------------------------------------------------------------
        let legend: Option<Vec<(String, String)>> = if config.color_by.is_some() {
            Some(
                series_labels
                    .iter()
                    .filter(|(label, _)| !label.is_empty())
                    .cloned()
                    .collect(),
            )
        } else {
            None
        };

        // Include "null" series in the legend if it had any data.
        let legend = legend.map(|mut entries| {
            let has_null = series_labels.iter().any(|(l, _)| l == "null")
                && data
                    .iter()
                    .zip(series_labels.iter())
                    .any(|(series, (label, _))| label == "null" && series.iter().any(|v| v.is_some()));
            if has_null {
                let null_color = series_labels
                    .iter()
                    .find(|(l, _)| l == "null")
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(|| NULL_COLOR.to_string());
                if !entries.iter().any(|(l, _)| l == "null") {
                    entries.push(("null".to_string(), null_color));
                }
            }
            entries
        });

        // ------------------------------------------------------------------
        // 11. Assemble Chart
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
// Rendering helper
// ---------------------------------------------------------------------------

/// Renders a single bar rectangle given the categorical-axis start/width and
/// the value-axis base/top pixel coordinates.
///
/// For vertical orientation: categorical axis = x, value axis = y.
/// For horizontal orientation: value axis = x, categorical axis = y.
fn bar_rect(
    orientation: &Orientation,
    cat_start:  f64,
    cat_width:  f64,
    val_base:   f64,  // pixel coordinate at y=0 (the baseline)
    val_top:    f64,  // pixel coordinate at y=value (the bar top)
    color:      &str,
) -> String {
    match orientation {
        Orientation::Vertical => {
            // SVG y increases downward: top of bar is val_top (smaller px), bottom is val_base.
            let rect_y = val_top.min(val_base);
            let rect_h = (val_base - val_top).abs();
            geometry::rect(cat_start, rect_y, cat_width, rect_h, color, 0.0)
        }
        Orientation::Horizontal => {
            // Bars extend rightward from the baseline (val_base = ox for y=0).
            let rect_x = val_base.min(val_top);
            let rect_w = (val_top - val_base).abs();
            geometry::rect(rect_x, cat_start, rect_w, cat_width, color, 0.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// Geometry reference (Margin top=50 right=30 bottom=60 left=70, 800×500):
//   ox=70  oy=50  pw=700  ph=390
//   plot_bottom = oy + ph = 440    plot_right = ox + pw = 770
//
// Fixture "simple_df"  cats=[A,B,C]  vals=[10,25,15]
//   nice_ticks(0,25,6) → [0,5,10,15,20,25]  domain=[0,25]
//   val_scale vertical: map(v) = 440 - v*(390/25) = 440 - 15.6v
//   band_px=233.33  band_pad=35.00  bar_w=163.33
//   zero_px = 440
//   A: x=105.00  w=163.33  y=284.00  h=156.00
//   B: x=338.33  w=163.33  y= 50.00  h=390.00
//   C: x=571.67  w=163.33  y=206.00  h=234.00
//
// Fixture "simple_df" HORIZONTAL
//   val_scale horizontal: map(v)= 70 + v*(700/25) = 70 + 28v
//   band_px_h=130  band_pad_h=19.5  bar_h_dim=91
//   zero_px_h = 70
//   A: x=70  w=280  y=69.5   h=91
//   B: x=70  w=700  y=199.5  h=91
//   C: x=70  w=420  y=329.5  h=91
//
// Fixture "multiseries_df"  A=[X:10,Y:5]  B=[X:20,Y:8]  C=[X:15,Y:3]
//   Stacked y_max=28  nice_ticks(0,28,6) → [0..30]  map(v) = 440 - 13v
//   A_X: y=310  h=130  A_Y: y=245  h= 65  (A_Y bottom = 310 = A_X top)
//   B_X: y=180  h=260  B_Y: y= 76  h=104  (B_Y bottom = 180 = B_X top)
//   C_X: y=245  h=195  C_Y: y=206  h= 39
//
// Null-y fixture  cats=[A,B,C]  vals=[10,null,15]
//   y_max=15  nice_ticks(0,15,6) → [0,2,4,6,8,10,12,14,16]  domain=[0,16]
//   zero_px=440   A: h=243.75   C: h=365.625
//
// Null-x fixture  rows=[(A,10),(null,5),(B,20)]
//   y_max=20  nice_ticks(0,20,6) → [0,5,10,15,20]  domain=[0,20]
//   map(v) = 440 - v*(390/20) = 440 - 19.5v
//   A: x=105  h=195   null: x=338.33  h=97.5   B: x=571.67  h=390

// ── Shared SVG-parsing utilities ─────────────────────────────────────────────

#[cfg(test)]
mod svg_util {
    /// Parse the numeric value of `attr="…"` from a single `<rect …/>` token.
    pub fn attr(rect: &str, name: &str) -> f64 {
        let needle = format!("{name}=\"");
        let s = rect.find(&needle).unwrap_or_else(|| panic!("attr '{name}' missing in: {rect}"));
        let after = &rect[s + needle.len()..];
        let end   = after.find('"').expect("closing quote");
        after[..end].parse::<f64>().unwrap_or_else(|_| panic!("non-numeric attr '{name}'"))
    }

    /// Return the `fill="…"` string from a single `<rect …/>` token.
    pub fn fill(rect: &str) -> &str {
        let s = rect.find("fill=\"").expect("fill attr") + 6;
        let e = rect[s..].find('"').expect("fill closing quote");
        &rect[s..s + e]
    }

    /// Extract all `<rect …/>` tokens whose width and height are both < 700
    /// (excludes the full-canvas background rect).
    /// Extract all data `<rect …/>` tokens — bars, histogram bins, heatmap cells, etc.
    ///
    /// Excludes:
    /// - The canvas background rect (full canvas dimensions)
    /// - The clipPath definition rect inside `<defs>…</defs>`
    /// - Legend swatches (carry an `rx` attribute)
    pub fn data_rects(svg: &str) -> Vec<String> {
        // Strip <defs>…</defs> so the clipPath rect is never scanned.
        let stripped: String = if let (Some(ds), Some(de)) =
            (svg.find("<defs>"), svg.find("</defs>"))
        {
            format!("{}{}", &svg[..ds], &svg[de + 7..])
        } else {
            svg.to_string()
        };

        let mut rects = Vec::new();
        let mut rest  = stripped.as_str();
        while let Some(start) = rest.find("<rect ") {
            rest = &rest[start..];
            let end = rest.find("/>").expect("unclosed <rect");
            let tok = &rest[..end + 2];
            let w = tok.find("width=\"")
                .map(|i| tok[i+7..].split('"').next().unwrap_or("0").parse::<f64>().unwrap_or(0.0))
                .unwrap_or(0.0);
            let h = tok.find("height=\"")
                .map(|i| tok[i+8..].split('"').next().unwrap_or("0").parse::<f64>().unwrap_or(0.0))
                .unwrap_or(0.0);
            // Skip canvas background (800×500) and legend swatches (rx attribute)
            if w < 800.0 && h < 500.0 && !tok.contains("rx=\"") {
                rects.push(tok.to_string());
            }
            rest = &rest[end + 2..];
        }
        rects
    }

    /// Count `<rect … rx="…"/>` elements — these are legend swatches.
    pub fn legend_swatch_count(svg: &str) -> usize {
        svg.matches("rx=\"").count()
    }
}

// ── 2.4.5a  Basic vertical bar chart ─────────────────────────────────────────

#[cfg(test)]
mod test_vertical {
    use super::*;
    use super::svg_util::*;
    use polars::prelude::*;

    fn simple_df() -> DataFrame {
        df!("category" => &["A","B","C"],
            "value"    => &[10.0f64, 25.0, 15.0]).unwrap()
    }

    #[test]
    fn svg_is_well_formed() {
        let df  = simple_df();
        let svg = BarBuilder::new(&df).x("category").y("value")
            .build().unwrap().svg().to_string();
        assert!(svg.starts_with("<svg"),  "must start with <svg");
        assert!(svg.ends_with("</svg>"), "must end with </svg>");
    }

    #[test]
    fn canvas_dimensions_are_800_x_500() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        assert_eq!(chart.width(),  800);
        assert_eq!(chart.height(), 500);
    }

    #[test]
    fn clean_data_produces_no_warnings() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        assert!(chart.warnings().is_empty(), "warnings: {:?}", chart.warnings());
    }

    #[test]
    fn three_categories_produce_exactly_three_data_rects() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 3, "expected 3 data rects, got {}: {rects:#?}", rects.len());
    }

    #[test]
    fn category_labels_appear_as_axis_text() {
        let df  = simple_df();
        let svg = BarBuilder::new(&df).x("category").y("value").build().unwrap().svg().to_string();
        assert!(svg.contains(">A<") || svg.contains("A ") || svg.contains(">A "), "A missing");
        assert!(svg.contains(">B<") || svg.contains("B ") || svg.contains(">B "), "B missing");
        assert!(svg.contains(">C<") || svg.contains("C ") || svg.contains(">C "), "C missing");
    }

    // ── Pixel-level geometry: all bottoms equal zero_px=440 ──────────────

    #[test]
    fn all_bars_share_the_same_baseline() {
        // Every bar must extend down to zero_px = oy+ph = 440.
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 3);
        for (i, r) in rects.iter().enumerate() {
            let bottom = attr(r, "y") + attr(r, "height");
            assert!(
                (bottom - 440.0).abs() < 0.5,
                "bar {i} bottom must be 440 (zero_px); got {bottom:.3}"
            );
        }
    }

    // ── A: x=105, w=163.33, y=284, h=156 ────────────────────────────────

    #[test]
    fn bar_a_has_correct_x_and_width() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        let x = attr(&rects[0], "x");
        let w = attr(&rects[0], "width");
        assert!((x - 105.0).abs()    < 0.5, "A x expected 105.0, got {x:.3}");
        assert!((w - 163.333).abs()  < 0.5, "A w expected 163.33, got {w:.3}");
    }

    #[test]
    fn bar_a_has_correct_height_for_value_10() {
        // nice_ticks(0,25) → domain [0,25]; map(10) = 440 - 10*(390/25) = 440-156 = 284 → h=156
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        let h = attr(&rects[0], "height");
        assert!((h - 156.0).abs() < 0.5, "A height expected 156.0, got {h:.3}");
    }

    // ── B: tallest bar (value=25) sits exactly at the top of the plot ─────

    #[test]
    fn bar_b_top_reaches_the_top_of_the_plot_area() {
        // B value=25 = domain max; map(25) = 440-390 = 50 = oy → h=390
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        let y_b = attr(&rects[1], "y");
        let h_b = attr(&rects[1], "height");
        assert!((y_b - 50.0).abs()  < 0.5, "B top (y) expected 50.0, got {y_b:.3}");
        assert!((h_b - 390.0).abs() < 0.5, "B height expected 390.0, got {h_b:.3}");
    }

    // ── C: x=571.67, h=234 ───────────────────────────────────────────────

    #[test]
    fn bar_c_has_correct_x_and_height_for_value_15() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        let x = attr(&rects[2], "x");
        let h = attr(&rects[2], "height");
        assert!((x - 571.667).abs() < 0.5, "C x expected 571.67, got {x:.3}");
        assert!((h - 234.0).abs()   < 0.5, "C height expected 234.0, got {h:.3}");
    }

    // ── Height ordering must match value ordering ─────────────────────────

    #[test]
    fn bar_heights_are_ordered_by_value() {
        // B(25) > C(15) > A(10)
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        let [h_a, h_b, h_c] = [
            attr(&rects[0], "height"),
            attr(&rects[1], "height"),
            attr(&rects[2], "height"),
        ];
        assert!(h_b > h_c && h_c > h_a,
            "expected h_B > h_C > h_A; got {h_b:.2}, {h_c:.2}, {h_a:.2}");
    }

    // ── Bars must not overlap each other horizontally ─────────────────────

    #[test]
    fn bars_do_not_overlap_horizontally() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        let rects = data_rects(chart.svg());
        // right edge of bar[i] ≤ left edge of bar[i+1]
        for i in 0..rects.len()-1 {
            let right_i = attr(&rects[i],   "x") + attr(&rects[i],   "width");
            let left_j  = attr(&rects[i+1], "x");
            assert!(right_i <= left_j + 0.1,
                "bar {i} right ({right_i:.2}) overlaps bar {} left ({left_j:.2})", i+1);
        }
    }

    // ── All bars must stay inside the plot area ───────────────────────────

    #[test]
    fn bars_are_within_plot_area() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value").build().unwrap();
        for r in data_rects(chart.svg()) {
            let x = attr(&r, "x"); let w = attr(&r, "width");
            let y = attr(&r, "y"); let h = attr(&r, "height");
            assert!(x      >= 70.0  - 0.1, "x={x:.2} < ox=70");
            assert!(x + w  <= 770.0 + 0.1, "right={:.2} > ox+pw=770", x+w);
            assert!(y      >= 50.0  - 0.1, "y={y:.2} < oy=50");
            assert!(y + h  <= 440.0 + 0.1, "bottom={:.2} > oy+ph=440", y+h);
        }
    }

    // ── Value axis must include tick "0" ──────────────────────────────────

    #[test]
    fn value_axis_has_zero_tick() {
        let df  = simple_df();
        let svg = BarBuilder::new(&df).x("category").y("value").build().unwrap().svg().to_string();
        assert!(svg.contains(">0<"), "tick label '0' missing from value axis");
    }

    // ── Single-row dataset must not panic ─────────────────────────────────

    #[test]
    fn single_row_builds_without_panic() {
        let df = df!("cat" => &["Solo"], "val" => &[42.0f64]).unwrap();
        BarBuilder::new(&df).x("cat").y("val").build().expect("single row must build");
    }
}

// ── 2.4.5b  Horizontal orientation — axis swap ───────────────────────────────

#[cfg(test)]
mod test_horizontal {
    use super::*;
    use super::svg_util::*;
    use polars::prelude::*;

    fn simple_df() -> DataFrame {
        df!("category" => &["A","B","C"],
            "value"    => &[10.0f64, 25.0, 15.0]).unwrap()
    }

    #[test]
    fn horizontal_chart_builds_without_error() {
        let df = simple_df();
        BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build()
            .expect("horizontal bar chart must build");
    }

    #[test]
    fn horizontal_produces_exactly_three_data_rects() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 3, "expected 3 data rects, got {}", rects.len());
    }

    #[test]
    fn canvas_dimensions_match_vertical() {
        let df  = simple_df();
        let v   = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Vertical).build().unwrap();
        let h   = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        assert_eq!(v.width(),  h.width());
        assert_eq!(v.height(), h.height());
    }

    // ── All bars start at zero_px_h = ox = 70 (left edge) ────────────────

    #[test]
    fn all_bars_start_at_left_baseline() {
        // Horizontal zero_px = val_scale(0) = ox = 70
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 3);
        for (i, r) in rects.iter().enumerate() {
            let x = attr(r, "x");
            assert!(
                (x - 70.0).abs() < 0.5,
                "horizontal bar {i} x must be 70 (zero_px_h); got {x:.3}"
            );
        }
    }

    // ── A: x=70, w=280, y=69.5, h=91 ────────────────────────────────────

    #[test]
    fn bar_a_has_correct_horizontal_geometry() {
        // nice_ticks(0,25) → domain [0,25]
        // val_scale h: map(10) = 70 + 10*(700/25) = 70+280 = 350 → w=350-70=280
        // band_px_h=130, band_pad=19.5, bar_h_dim=91
        // A: band_cy = 50 + (1/6)*390 = 50+65 = 115
        //    bar_start_y = 115 - 65 + 19.5 = 69.5
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        let w = attr(&rects[0], "width");
        let y = attr(&rects[0], "y");
        let h = attr(&rects[0], "height");
        assert!((w - 280.0).abs() < 0.5, "A width expected 280.0, got {w:.3}");
        assert!((y -  69.5).abs() < 0.5, "A y expected 69.5, got {y:.3}");
        assert!((h -  91.0).abs() < 0.5, "A height expected 91.0, got {h:.3}");
    }

    // ── B: widest bar (value=25) spans the full plot width ────────────────

    #[test]
    fn bar_b_spans_full_plot_width() {
        // map(25) = 70 + 25*(700/25) = 70+700 = 770 → w = 770-70 = 700
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        let w_b = attr(&rects[1], "width");
        assert!((w_b - 700.0).abs() < 0.5, "B width expected 700.0, got {w_b:.3}");
    }

    // ── C: w=420, y=329.5, h=91 ──────────────────────────────────────────

    #[test]
    fn bar_c_has_correct_horizontal_geometry() {
        // map(15) = 70 + 15*(700/25) = 70+420 = 490 → w=420
        // C: band_cy = 50 + (5/6)*390 = 50+325 = 375
        //    bar_start_y = 375 - 65 + 19.5 = 329.5
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        let w = attr(&rects[2], "width");
        let y = attr(&rects[2], "y");
        let h = attr(&rects[2], "height");
        assert!((w - 420.0).abs() < 0.5, "C width expected 420.0, got {w:.3}");
        assert!((y - 329.5).abs() < 0.5, "C y expected 329.5, got {y:.3}");
        assert!((h -  91.0).abs() < 0.5, "C height expected 91.0, got {h:.3}");
    }

    // ── Width ordering matches value ordering ─────────────────────────────

    #[test]
    fn bar_widths_are_ordered_by_value() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        let [w_a, w_b, w_c] = [
            attr(&rects[0], "width"),
            attr(&rects[1], "width"),
            attr(&rects[2], "width"),
        ];
        assert!(w_b > w_c && w_c > w_a,
            "expected w_B > w_C > w_A; got {w_b:.2}, {w_c:.2}, {w_a:.2}");
    }

    // ── All bars share the same height (band height) ──────────────────────

    #[test]
    fn all_bars_share_the_same_band_height() {
        // band_h_dim = 130 - 2*19.5 = 91 for every bar
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        let heights: Vec<f64> = rects.iter().map(|r| attr(r, "height")).collect();
        for (i, &h) in heights.iter().enumerate() {
            assert!((h - 91.0).abs() < 0.5,
                "horizontal bar {i} height expected 91.0, got {h:.3}");
        }
    }

    // ── Bars must not overlap each other vertically ───────────────────────

    #[test]
    fn bars_do_not_overlap_vertically() {
        let df    = simple_df();
        let chart = BarBuilder::new(&df).x("category").y("value")
            .orientation(Orientation::Horizontal).build().unwrap();
        let rects = data_rects(chart.svg());
        for i in 0..rects.len()-1 {
            let bottom_i = attr(&rects[i],   "y") + attr(&rects[i],   "height");
            let top_j    = attr(&rects[i+1], "y");
            assert!(bottom_i <= top_j + 0.1,
                "horizontal bar {i} bottom ({bottom_i:.2}) overlaps bar {} top ({top_j:.2})", i+1);
        }
    }

    // ── Axis swap evidence: vertical heights ≠ horizontal heights ────────

    #[test]
    fn value_dimension_is_width_not_height_in_horizontal_mode() {
        // In vertical mode the value axis drives height.
        // In horizontal mode the value axis drives width.
        // Check: vertical heights are NOT uniform (values differ) while
        //        horizontal heights ARE uniform (all = band_h_dim).
        let df = simple_df();
        let v_rects = data_rects(
            BarBuilder::new(&df).x("category").y("value")
                .orientation(Orientation::Vertical).build().unwrap().svg()
        );
        let h_rects = data_rects(
            BarBuilder::new(&df).x("category").y("value")
                .orientation(Orientation::Horizontal).build().unwrap().svg()
        );
        // Vertical: heights vary
        let v_heights: Vec<f64> = v_rects.iter().map(|r| attr(r, "height")).collect();
        assert!(v_heights[0] != v_heights[1], "vertical heights should differ");
        // Horizontal: heights are uniform (all band_h_dim=91)
        let h_heights: Vec<f64> = h_rects.iter().map(|r| attr(r, "height")).collect();
        assert!((h_heights[0] - h_heights[1]).abs() < 0.5,
            "horizontal bar heights should be equal (same band); got {:?}", h_heights);
        // Horizontal: widths vary (driven by value)
        let h_widths: Vec<f64> = h_rects.iter().map(|r| attr(r, "width")).collect();
        assert!(h_widths[0] != h_widths[1], "horizontal widths should differ by value");
    }
}

// ── 2.4.5c  Stacked bars — cumulative heights ────────────────────────────────

#[cfg(test)]
mod test_stacked {
    use super::*;
    use super::svg_util::*;
    use polars::prelude::*;

    // A=[X:10,Y:5]  B=[X:20,Y:8]  C=[X:15,Y:3]
    // y_max_cumulative = 28  nice_ticks(0,28,6) → [0..30]  domain=[0,30]
    // map(v) = 440 - v*(390/30) = 440 - 13v
    fn multi_df() -> DataFrame {
        df!(
            "category" => &["A","A","B","B","C","C"],
            "value"    => &[10.0f64, 5.0, 20.0, 8.0, 15.0, 3.0],
            "group"    => &["X","Y","X","Y","X","Y"]
        ).unwrap()
    }

    #[test]
    fn stacked_chart_builds_without_error() {
        let df = multi_df();
        BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().expect("stacked chart must build");
    }

    #[test]
    fn stacked_3_categories_2_series_gives_6_rects() {
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 6,
            "3 cats × 2 series = 6 data rects; got {}", rects.len());
    }

    // ── Category A: X bottom=440, X top=310; Y bottom=310, Y top=245 ─────

    #[test]
    fn category_a_x_segment_has_correct_rect() {
        // A_X: val=10  base=map(0)=440  top=map(10)=310  → y=310, h=130
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        // Emission order: A_X, A_Y, B_X, B_Y, C_X, C_Y
        let y = attr(&rects[0], "y");
        let h = attr(&rects[0], "height");
        assert!((y - 310.0).abs() < 0.5, "A_X y expected 310.0, got {y:.3}");
        assert!((h - 130.0).abs() < 0.5, "A_X h expected 130.0, got {h:.3}");
    }

    #[test]
    fn category_a_y_segment_sits_directly_atop_x_segment() {
        // A_Y: val=5  base=map(10)=310  top=map(15)=245  → y=245, h=65
        // A_Y bottom (y+h=310) must equal A_X top (y=310)
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        let y_y = attr(&rects[1], "y");
        let h_y = attr(&rects[1], "height");
        let y_x = attr(&rects[0], "y"); // A_X top
        // A_Y bottom = y_y + h_y must equal A_X y (the top of A_X)
        let a_y_bottom = y_y + h_y;
        assert!((a_y_bottom - y_x).abs() < 0.5,
            "A_Y bottom ({a_y_bottom:.3}) must equal A_X top ({y_x:.3})");
        assert!((y_y - 245.0).abs() < 0.5, "A_Y y expected 245.0, got {y_y:.3}");
        assert!((h_y -  65.0).abs() < 0.5, "A_Y h expected  65.0, got {h_y:.3}");
    }

    // ── Category B: X→y=180,h=260; Y→y=76,h=104 ─────────────────────────

    #[test]
    fn category_b_x_segment_has_correct_rect() {
        // B_X: val=20  base=440  top=map(20)=180  → y=180, h=260
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        let y = attr(&rects[2], "y");
        let h = attr(&rects[2], "height");
        assert!((y - 180.0).abs() < 0.5, "B_X y expected 180.0, got {y:.3}");
        assert!((h - 260.0).abs() < 0.5, "B_X h expected 260.0, got {h:.3}");
    }

    #[test]
    fn category_b_y_segment_has_correct_rect_and_touches_x_segment() {
        // B_Y: val=8  base=map(20)=180  top=map(28)=440-364=76  → y=76, h=104
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        let y_y = attr(&rects[3], "y");
        let h_y = attr(&rects[3], "height");
        let y_x = attr(&rects[2], "y"); // B_X top
        let b_y_bottom = y_y + h_y;
        assert!((y_y -  76.0).abs() < 0.5,  "B_Y y expected  76.0, got {y_y:.3}");
        assert!((h_y - 104.0).abs() < 0.5,  "B_Y h expected 104.0, got {h_y:.3}");
        assert!((b_y_bottom - y_x).abs() < 0.5,
            "B_Y bottom ({b_y_bottom:.3}) must touch B_X top ({y_x:.3})");
    }

    // ── All stack bottoms touch zero_px=440 ──────────────────────────────

    #[test]
    fn every_bottom_segment_starts_at_zero_px() {
        // In each category the X segment (emitted first) must have bottom=440.
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        for (cat_idx, rect_idx) in [(0,0), (1,2), (2,4)] {
            let y = attr(&rects[rect_idx], "y");
            let h = attr(&rects[rect_idx], "height");
            let bottom = y + h;
            assert!((bottom - 440.0).abs() < 0.5,
                "cat {cat_idx} bottom segment bottom must be 440; got {bottom:.3}");
        }
    }

    // ── Stacked axis covers the cumulative max ────────────────────────────

    #[test]
    fn value_axis_covers_cumulative_max_of_28() {
        // nice_ticks(0,28,6) → last tick = 30; "30" must appear in the SVG
        let df  = multi_df();
        let svg = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap().svg().to_string();
        assert!(svg.contains(">30<") || svg.contains(">28<") || svg.contains(">25<"),
            "stacked axis must tick past the cumulative max of 28");
        // At minimum the axis must not stop at the grouped max (20)
        assert!(!svg.contains(">20<") || svg.contains(">25<") || svg.contains(">30<"),
            "stacked axis must extend beyond the individual max of 20");
    }

    // ── Grouped axis does NOT reach 30 ────────────────────────────────────

    #[test]
    fn grouped_axis_stops_at_individual_max() {
        // nice_ticks(0,20,6) → [0,5,10,15,20]  — no tick at 30
        let df  = multi_df();
        let svg = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(false).build().unwrap().svg().to_string();
        // "30" must not appear as a tick label in grouped mode
        assert!(!svg.contains(">30<"),
            "grouped axis must not reach 30 (individual max is 20)");
    }

    // ── Stacked segments within a category share the same x and width ─────

    #[test]
    fn stacked_segments_in_same_category_share_x_and_width() {
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        // Pairs: (A_X,A_Y)=(0,1), (B_X,B_Y)=(2,3), (C_X,C_Y)=(4,5)
        for (i, j) in [(0usize,1usize),(2,3),(4,5)] {
            let x0 = attr(&rects[i], "x"); let x1 = attr(&rects[j], "x");
            let w0 = attr(&rects[i], "width"); let w1 = attr(&rects[j], "width");
            assert!((x0-x1).abs() < 0.5,
                "stacked pair ({i},{j}) must share x; got {x0:.2} vs {x1:.2}");
            assert!((w0-w1).abs() < 0.5,
                "stacked pair ({i},{j}) must share width; got {w0:.2} vs {w1:.2}");
        }
    }

    // ── Height ratio within A must match value ratio 10:5 = 2:1 ──────────

    #[test]
    fn segment_heights_are_proportional_to_values() {
        // A_X(10) / A_Y(5) = 2.0  →  h_X / h_Y = 130/65 = 2.0
        let df    = multi_df();
        let chart = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        let ratio = attr(&rects[0], "height") / attr(&rects[1], "height");
        assert!((ratio - 2.0).abs() < 0.05,
            "A_X/A_Y height ratio must be 2.0 (values 10 and 5); got {ratio:.4}");
    }

    // ── Legend contains both group labels ──────────────────────────────────

    #[test]
    fn legend_contains_both_series_labels() {
        let df  = multi_df();
        let svg = BarBuilder::new(&df).x("category").y("value").color_by("group")
            .stacked(true).build().unwrap().svg().to_string();
        assert!(svg.contains("X"), "legend missing 'X'");
        assert!(svg.contains("Y"), "legend missing 'Y'");
    }

    // ── stacked(true) without color_by falls through to single-series ─────

    #[test]
    fn stacked_without_color_by_behaves_like_simple_chart() {
        // n_series=1 → the stacked branch (`n_series > 1`) is never entered.
        let df = df!("cat" => &["A","B","C"], "val" => &[10.0f64,20.0,15.0]).unwrap();
        let stacked = data_rects(
            BarBuilder::new(&df).x("cat").y("val").stacked(true).build().unwrap().svg()
        ).len();
        let simple  = data_rects(
            BarBuilder::new(&df).x("cat").y("val").build().unwrap().svg()
        ).len();
        assert_eq!(stacked, simple,
            "stacked without color_by must equal simple chart; stacked={stacked}, simple={simple}");
    }
}

// ── 2.4.4a  Null x renders as the category "null" ────────────────────────────

#[cfg(test)]
mod test_null_x {
    use super::*;
    use super::svg_util::*;
    use polars::prelude::*;

    // rows=[(A,10),(null,5),(B,20)]  → cats=[A,null,B]  y_max=20
    // nice_ticks(0,20,6) → [0,5,10,15,20]  domain=[0,20]
    // map(v) = 440 - v*(390/20) = 440 - 19.5v
    // band_px=233.33  bar_w=163.33  zero_px=440
    // A:    x=105,    h=195   (map(10)=245, h=195)
    // null: x=338.33, h=97.5  (map(5) =342.5, h=97.5)
    // B:    x=571.67, h=390   (map(20)=50, h=390)

    fn null_x_df() -> DataFrame {
        df!("cat" => &[Some("A"), None, Some("B")],
            "val" => &[10.0f64, 5.0, 20.0]).unwrap()
    }

    #[test]
    fn null_x_chart_builds_without_error() {
        let df = null_x_df();
        BarBuilder::new(&df).x("cat").y("val").build().expect("null x must build");
    }

    #[test]
    fn null_x_produces_three_data_rects() {
        // Three rows, three distinct categories (A, "null", B) → 3 bars.
        let df    = null_x_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 3,
            "A/null/B should produce 3 bars; got {}", rects.len());
    }

    #[test]
    fn null_category_label_appears_in_svg() {
        let df  = null_x_df();
        let svg = BarBuilder::new(&df).x("cat").y("val").build().unwrap().svg().to_string();
        assert!(svg.contains("null"), "'null' category label must appear in the SVG");
    }

    #[test]
    fn null_bar_has_correct_x_position() {
        // "null" is the second category (index 1): bar_start = 338.33
        let df    = null_x_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        let x = attr(&rects[1], "x");
        assert!((x - 338.333).abs() < 0.5, "null bar x expected 338.33, got {x:.3}");
    }

    #[test]
    fn null_bar_has_correct_height_for_value_5() {
        // map(5) = 440 - 5*19.5 = 440 - 97.5 = 342.5  → h = 97.5
        let df    = null_x_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        let h = attr(&rects[1], "height");
        assert!((h - 97.5).abs() < 0.5, "null bar h expected 97.5, got {h:.3}");
    }

    #[test]
    fn null_bar_touches_the_same_baseline_as_other_bars() {
        let df    = null_x_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        for (i, r) in rects.iter().enumerate() {
            let bottom = attr(r, "y") + attr(r, "height");
            assert!((bottom - 440.0).abs() < 0.5,
                "bar {i} bottom must be 440; got {bottom:.3}");
        }
    }

    #[test]
    fn null_bar_is_shorter_than_a_and_b() {
        // null(5) < A(10) < B(20)
        let df    = null_x_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        let h_a    = attr(&rects[0], "height");
        let h_null = attr(&rects[1], "height");
        let h_b    = attr(&rects[2], "height");
        assert!(h_null < h_a, "null(5) must be shorter than A(10); h_null={h_null:.2}, h_a={h_a:.2}");
        assert!(h_null < h_b, "null(5) must be shorter than B(20); h_null={h_null:.2}, h_b={h_b:.2}");
    }

    #[test]
    fn two_null_x_rows_merge_into_one_null_category() {
        // rows = [(A,10), (null,3), (null,7)]  → categories = [A, "null"]
        // null bar aggregates: 3+7=10 → same height as A
        let df = df!("cat" => &[Some("A"), None, None],
                     "val" => &[10.0f64, 3.0, 7.0]).unwrap();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 2, "two null-x rows must merge into one bar; got {}", rects.len());
        let h_a    = attr(&rects[0], "height");
        let h_null = attr(&rects[1], "height");
        assert!((h_a - h_null).abs() < 0.5,
            "null(3+7=10) and A(10) must have equal heights; h_a={h_a:.2}, h_null={h_null:.2}");
    }
}

// ── 2.4.4b  Null y leaves a gap in the series ────────────────────────────────

#[cfg(test)]
mod test_null_y {
    use super::*;
    use super::svg_util::*;
    use polars::prelude::*;

    // cats=[A,B,C]  vals=[10,null,15]  y_max=15
    // nice_ticks(0,15,6) → [0,2,4,6,8,10,12,14,16]  domain=[0,16]
    // map(v) = 440 - v*(390/16) = 440 - 24.375v
    // A: h=243.75  C: h=365.625  (B: absent)

    fn null_y_df() -> DataFrame {
        df!("cat" => &["A","B","C"],
            "val" => &[Some(10.0f64), None, Some(15.0)]).unwrap()
    }

    #[test]
    fn null_y_chart_builds_without_error() {
        let df = null_y_df();
        BarBuilder::new(&df).x("cat").y("val").build().expect("null y must build");
    }

    #[test]
    fn null_y_bar_is_absent_leaving_two_rects_not_three() {
        // B has null y → B gets no rect; only A and C bars render.
        let df    = null_y_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 2,
            "null y for B must leave a gap (expect 2 rects); got {}", rects.len());
    }

    #[test]
    fn null_y_emits_one_nulls_skipped_warning() {
        let df    = null_y_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let warned = chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, count: 1 } if col == "val"
        ));
        assert!(warned,
            "must emit NullsSkipped {{ col: \"val\", count: 1 }}; got {:?}",
            chart.warnings());
    }

    #[test]
    fn null_y_warning_count_matches_number_of_nulls() {
        // Three nulls → count = 3
        let df = df!("cat" => &["A","B","C","D","E"],
                     "val" => &[Some(1.0f64), None, None, Some(4.0), None]).unwrap();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let count: usize = chart.warnings().iter()
            .filter_map(|w| match w {
                CharcoalWarning::NullsSkipped { col, count } if col == "val" => Some(*count),
                _ => None,
            })
            .sum();
        assert_eq!(count, 3, "three null y rows must produce NullsSkipped count=3; got {count}");
    }

    #[test]
    fn null_y_two_remaining_bars_have_correct_heights() {
        // A: h = 10*(390/16) = 243.75   C: h = 15*(390/16) = 365.625
        let df    = null_y_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 2);
        let h_a = attr(&rects[0], "height");
        let h_c = attr(&rects[1], "height");
        assert!((h_a - 243.75).abs()  < 0.5, "A height expected 243.75, got {h_a:.3}");
        assert!((h_c - 365.625).abs() < 0.5, "C height expected 365.625, got {h_c:.3}");
    }

    #[test]
    fn null_y_remaining_bars_still_share_baseline() {
        let df    = null_y_df();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        for r in data_rects(chart.svg()) {
            let bottom = attr(&r, "y") + attr(&r, "height");
            assert!((bottom - 440.0).abs() < 0.5, "bar bottom must be 440; got {bottom:.3}");
        }
    }

    #[test]
    fn null_y_fewer_rects_than_all_present() {
        // Cross-check: a fully populated dataset produces more bars.
        let null_df = null_y_df();
        let full_df = df!("cat" => &["A","B","C"],
                          "val" => &[10.0f64, 5.0, 15.0]).unwrap();
        let null_n = data_rects(
            BarBuilder::new(&null_df).x("cat").y("val").build().unwrap().svg()
        ).len();
        let full_n = data_rects(
            BarBuilder::new(&full_df).x("cat").y("val").build().unwrap().svg()
        ).len();
        assert!(null_n < full_n,
            "null-y chart must have fewer rects than all-present; null={null_n}, full={full_n}");
    }

    #[test]
    fn all_null_y_for_category_leaves_complete_gap() {
        // B appears twice but both rows have null y → zero bars for B.
        let df = df!("cat" => &["A","B","B","C"],
                     "val" => &[Some(10.0f64), None, None, Some(5.0)]).unwrap();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 2,
            "all-null B must leave B entirely absent; got {} rects", rects.len());
    }

    #[test]
    fn null_y_in_stacked_series_leaves_gap_for_that_series() {
        // A_Y is null; only A_X renders for A. B has both X and Y → 2 rects.
        // Total = 1 + 2 = 3.
        let df = df!(
            "cat"   => &["A","A","B","B"],
            "val"   => &[Some(10.0f64), None, Some(20.0), Some(8.0)],
            "group" => &["X","Y","X","Y"]
        ).unwrap();
        let chart = BarBuilder::new(&df).x("cat").y("val").color_by("group")
            .stacked(true).build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 3,
            "A(X only) + B(X+Y) = 3 rects; got {}", rects.len());
    }

    #[test]
    fn null_x_combined_with_null_y_produces_no_bar() {
        // (null_x, null_y) → "null" category exists but has no non-null y → 0 bars for it.
        // Only A renders.
        let df = df!("cat" => &[Some("A"), None],
                     "val" => &[Some(10.0f64), None]).unwrap();
        let chart = BarBuilder::new(&df).x("cat").y("val").build().unwrap();
        let rects = data_rects(chart.svg());
        assert_eq!(rects.len(), 1,
            "null_x+null_y row must contribute no bar; got {} rects", rects.len());
    }
}

// ── Column-role validation ────────────────────────────────────────────────────

#[cfg(test)]
mod test_column_validation {
    use super::*;
    use polars::prelude::*;

    fn cat_num() -> DataFrame {
        df!("category" => &["A","B","C"],
            "value"    => &[10.0f64, 25.0, 15.0]).unwrap()
    }

    #[test]
    fn float_x_returns_unsupported_column_naming_x() {
        let df  = df!("num" => &[1.0f64,2.0,3.0], "val" => &[10.0f64,20.0,30.0]).unwrap();
        let err = BarBuilder::new(&df).x("num").y("val").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { ref col, .. } if col == "num"),
            "expected UnsupportedColumn for 'num', got {err:?}");
    }

    #[test]
    fn int_x_returns_unsupported_column() {
        let df  = df!("id" => &[1i64,2,3], "val" => &[10.0f64,20.0,30.0]).unwrap();
        let err = BarBuilder::new(&df).x("id").y("val").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { .. }));
    }

    #[test]
    fn numeric_x_error_mentions_categorical() {
        let df  = df!("num" => &[1.0f64,2.0,3.0], "val" => &[10.0f64,20.0,30.0]).unwrap();
        let err = BarBuilder::new(&df).x("num").y("val").build().unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(msg.contains("categorical"),
            "error must mention 'categorical'; got: {msg}");
    }

    #[test]
    fn string_y_returns_unsupported_column_naming_y() {
        let df  = df!("cat" => &["A","B"], "grp" => &["X","Y"]).unwrap();
        let err = BarBuilder::new(&df).x("cat").y("grp").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { ref col, .. } if col == "grp"),
            "expected UnsupportedColumn for 'grp', got {err:?}");
    }

    #[test]
    fn x_typo_returns_column_not_found_with_suggestion() {
        let df  = cat_num();
        let err = BarBuilder::new(&df).x("categori").y("value").build().unwrap_err();
        match err {
            CharcoalError::ColumnNotFound { ref name, ref suggestion, .. } => {
                assert_eq!(name, "categori");
                assert_eq!(suggestion, "category");
            }
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }

    #[test]
    fn y_typo_returns_column_not_found_with_suggestion() {
        let df  = cat_num();
        let err = BarBuilder::new(&df).x("category").y("vlue").build().unwrap_err();
        match err {
            CharcoalError::ColumnNotFound { ref name, ref suggestion, .. } => {
                assert_eq!(name, "vlue");
                assert_eq!(suggestion, "value");
            }
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }

    #[test]
    fn row_limit_exceeded_returns_data_too_large() {
        let df  = cat_num();   // 3 rows
        let err = BarBuilder::new(&df).x("category").y("value").row_limit(2).build().unwrap_err();
        match err {
            CharcoalError::DataTooLarge { rows, limit, .. } => {
                assert_eq!(rows, 3); assert_eq!(limit, 2);
            }
            other => panic!("expected DataTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn row_limit_exactly_equal_to_row_count_does_not_error() {
        let df = cat_num();
        BarBuilder::new(&df).x("category").y("value").row_limit(3).build()
            .expect("limit == n_rows must succeed");
    }

    #[test]
    fn palette_cycles_without_panic_for_nine_groups() {
        let labels: Vec<&str> = vec!["a","b","c","d","e","f","g","h","i"];
        let df = df!(
            "cat"   => &labels,
            "val"   => &vec![1.0f64; 9],
            "group" => &labels
        ).unwrap();
        let chart = BarBuilder::new(&df).x("cat").y("val").color_by("group")
            .build().expect("9 groups must not panic");
        for label in &labels {
            assert!(chart.svg().contains(label), "legend missing '{label}'");
        }
    }
}