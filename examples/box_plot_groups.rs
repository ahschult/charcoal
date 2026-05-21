use charcoal::{Chart, Theme};
use charcoal::PointDisplay;
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::new(vec![
        Series::new("group", &[
            "A", "A", "A", "A", "A", "A", "A", "A",
            "B", "B", "B", "B", "B", "B", "B", "B",
            "C", "C", "C", "C", "C", "C", "C", "C",
        ]),
        Series::new("value", &[
            2.1f64, 2.5, 2.8, 3.1, 3.3, 3.5, 3.8, 8.0,
            1.5f64, 1.9, 2.2, 2.6, 2.9, 3.1, 3.4, 3.7,
            3.0f64, 3.4, 3.7, 4.0, 4.2, 4.5, 4.8, 9.5,
        ]),
    ])?;

    let chart = Chart::box_plot(&df)
        .x("group")
        .y("value")
        .title("Value Distribution by Group")
        .notched(true)
        .points(PointDisplay::Outliers)
        .theme(Theme::Default)
        .build()?;

    chart.save_html("output.html")?;
    println!("box_plot_groups: saved output.html");
    Ok(())
}
