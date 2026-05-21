use charcoal::{Chart, ColorScale, Theme};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let labels = ["A", "B", "C", "D", "E"];
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut z_vals = Vec::new();

    let matrix: [[f64; 5]; 5] = [
        [1.00,  0.85,  0.60, -0.20, -0.50],
        [0.85,  1.00,  0.75,  0.10, -0.30],
        [0.60,  0.75,  1.00,  0.40,  0.05],
        [-0.20, 0.10,  0.40,  1.00,  0.65],
        [-0.50, -0.30, 0.05,  0.65,  1.00],
    ];

    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            x_vals.push(labels[j]);
            y_vals.push(labels[i]);
            z_vals.push(val);
        }
    }

    let df = DataFrame::new(vec![
        Series::new("x", &x_vals),
        Series::new("y", &y_vals),
        Series::new("correlation", &z_vals),
    ])?;

    let chart = Chart::heatmap(&df)
        .x("x")
        .y("y")
        .z("correlation")
        .title("Feature Correlation Matrix")
        .color_scale(ColorScale::Viridis)
        .annotate(true)
        .theme(Theme::Default)
        .build()?;

    chart.save_html("output.html")?;
    println!("heatmap_correlation: saved output.html");
    Ok(())
}
