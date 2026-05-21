use charcoal::{Chart, Theme};
use charcoal::FillMode;
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::new(vec![
        Series::new("week", &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Series::new("downloads", &[
            210.0f64, 240.0, 275.0, 310.0, 290.0, 330.0,
            150.0f64, 180.0, 195.0, 220.0, 210.0, 250.0,
        ]),
        Series::new("platform", &[
            "desktop", "desktop", "desktop", "desktop", "desktop", "desktop",
            "mobile", "mobile", "mobile", "mobile", "mobile", "mobile",
        ]),
    ])?;

    let chart = Chart::area(&df)
        .x("week")
        .y("downloads")
        .color_by("platform")
        .title("Weekly Downloads by Platform")
        .fill_mode(FillMode::ToZero)
        .stacked(true)
        .theme(Theme::Default)
        .build()?;

    chart.save_html("output.html")?;
    println!("area_stacked: saved output.html");
    Ok(())
}
