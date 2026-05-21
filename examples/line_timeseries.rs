use charcoal::{Chart, NullPolicy, Theme};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::new(vec![
        Series::new("month", &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Series::new("revenue", &[
            Some(120.0f64), Some(135.0), None, Some(160.0), Some(175.0), Some(190.0),
            Some(100.0f64), Some(115.0), Some(130.0), None, Some(155.0), Some(170.0),
        ]),
        Series::new("product", &[
            "widgets", "widgets", "widgets", "widgets", "widgets", "widgets",
            "gadgets", "gadgets", "gadgets", "gadgets", "gadgets", "gadgets",
        ]),
    ])?;

    let chart = Chart::line(&df)
        .x("month")
        .y("revenue")
        .color_by("product")
        .title("Monthly Revenue by Product")
        .null_policy(NullPolicy::Skip)
        .theme(Theme::Default)
        .build()?;

    chart.save_html("output.html")?;
    println!("line_timeseries: saved output.html");
    Ok(())
}
