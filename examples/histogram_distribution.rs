use charcoal::{Chart, Theme};
use charcoal::BinMethod;
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Approximation of a normal distribution centered at 0, std ~1
    let values: Vec<f64> = vec![
        -2.1, -1.8, -1.6, -1.5, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8,
        -0.7, -0.6, -0.5, -0.5, -0.4, -0.4, -0.3, -0.3, -0.2, -0.2,
        -0.1, -0.1,  0.0,  0.0,  0.0,  0.1,  0.1,  0.2,  0.2,  0.3,
         0.3,  0.4,  0.4,  0.5,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,
         1.1,  1.2,  1.3,  1.5,  1.6,  1.8,  2.1,  2.4, -0.6,  0.6,
    ];

    let df = DataFrame::new(vec![
        Series::new("value", &values),
    ])?;

    let chart = Chart::histogram(&df)
        .x("value")
        .title("Distribution of Values")
        .bins(20)
        .normalize(true)
        .bin_method(BinMethod::Scott)
        .theme(Theme::Default)
        .build()?;

    chart.save_html("output.html")?;
    println!("histogram_distribution: saved output.html");
    Ok(())
}
