use charcoal::{Chart, Orientation, Theme};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::new(vec![
        Series::new("category", &["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"]),
        Series::new("sales", &[42.0f64, 55.0, 61.0, 78.0, 38.0, 49.0, 57.0, 66.0]),
        Series::new("region", &["North", "North", "North", "North", "South", "South", "South", "South"]),
    ])?;

    let chart_vertical = Chart::bar(&df)
        .x("category")
        .y("sales")
        .color_by("region")
        .title("Quarterly Sales (Vertical)")
        .orientation(Orientation::Vertical)
        .theme(Theme::Default)
        .build()?;

    chart_vertical.save_html("output.html")?;
    println!("bar_categorical: saved output.html (vertical)");

    let chart_stacked = Chart::bar(&df)
        .x("category")
        .y("sales")
        .color_by("region")
        .title("Quarterly Sales (Stacked)")
        .stacked(true)
        .theme(Theme::Default)
        .build()?;

    chart_stacked.save_html("output_stacked.html")?;
    println!("bar_categorical: saved output_stacked.html (stacked)");

    Ok(())
}
