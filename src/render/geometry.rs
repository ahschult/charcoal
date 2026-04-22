#![allow(dead_code)]

fn fmt_f64(v: f64) -> String {
    format!("{:.2}", v)
}

pub(crate) fn circle(cx: f64, cy: f64, r: f64, fill: &str, opacity: f64) -> String {
    format!(
        r#"<circle cx="{}" cy="{}" r="{}" fill="{}" opacity="{}"/>"#,
        fmt_f64(cx),
        fmt_f64(cy),
        fmt_f64(r),
        fill,
        fmt_f64(opacity),
    )
}

pub(crate) fn rect(x: f64, y: f64, w: f64, h: f64, fill: &str, rx: f64) -> String {
    if rx == 0.0 {
        format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
            fmt_f64(x),
            fmt_f64(y),
            fmt_f64(w),
            fmt_f64(h),
            fill,
        )
    } else {
        format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="{}"/>"#,
            fmt_f64(x),
            fmt_f64(y),
            fmt_f64(w),
            fmt_f64(h),
            fill,
            fmt_f64(rx),
        )
    }
}

pub(crate) fn line(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: &str,
    width: f64,
) -> String {
    format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
        fmt_f64(x1),
        fmt_f64(y1),
        fmt_f64(x2),
        fmt_f64(y2),
        stroke,
        fmt_f64(width),
    )
}

pub(crate) fn dashed_line(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: &str,
    width: f64,
    dash: &str,
) -> String {
    let dash_attr = if dash.is_empty() {
        String::new()
    } else {
        format!(r#" stroke-dasharray="{}""#, dash)
    };
    format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"{}/>"#,
        fmt_f64(x1),
        fmt_f64(y1),
        fmt_f64(x2),
        fmt_f64(y2),
        stroke,
        fmt_f64(width),
        dash_attr,
    )
}

pub(crate) fn text(
    x: f64,
    y: f64,
    content: &str,
    anchor: &str,
    size: u32,
    color: &str,
    rotate: f64,
) -> String {
    let transform = if rotate == 0.0 {
        String::new()
    } else {
        format!(r#" transform="rotate({},{},{})" "#, fmt_f64(rotate), fmt_f64(x), fmt_f64(y))
    };

    format!(
        r#"<text x="{}" y="{}" text-anchor="{}" font-size="{}" fill="{}"{}>{}</text>"#,
        fmt_f64(x),
        fmt_f64(y),
        anchor,
        size,
        color,
        transform,
        escape_xml(content),
    )
}

pub(crate) fn points_to_path(points: &[(f64, f64)]) -> String {
    if points.is_empty() {
        return String::new();
    }
    let mut parts = Vec::with_capacity(points.len());
    let (x0, y0) = points[0];
    parts.push(format!("M {},{}", fmt_f64(x0), fmt_f64(y0)));
    for &(x, y) in &points[1..] {
        parts.push(format!("L {},{}", fmt_f64(x), fmt_f64(y)));
    }
    parts.join(" ")
}

pub(crate) fn polyline(points: &[(f64, f64)], stroke: &str, width: f64, dash: &str) -> String {
    if points.is_empty() {
        return String::new();
    }
    let path_data = points_to_path(points);
    let dash_attr = if dash.is_empty() {
        String::new()
    } else {
        format!(r#" stroke-dasharray="{}""#, dash)
    };
    format!(
        r#"<path d="{}" fill="none" stroke="{}" stroke-width="{}"{}/>"#,
        path_data,
        stroke,
        fmt_f64(width),
        dash_attr,
    )
}

pub(crate) fn polygon(
    points: &[(f64, f64)],
    fill: &str,
    stroke: &str,
    opacity: f64,
) -> String {
    if points.is_empty() {
        return String::new();
    }
    let path_data = format!("{} Z", points_to_path(points));
    format!(
        r#"<path d="{}" fill="{}" stroke="{}" opacity="{}"/>"#,
        path_data,
        fill,
        stroke,
        fmt_f64(opacity),
    )
}

pub(crate) fn group(elements: &[String], clip_id: Option<&str>) -> String {
    let clip_attr = match clip_id {
        Some(id) => format!(r#" clip-path="url(#{})""#, id),
        None => String::new(),
    };
    let inner = elements.join("\n");
    format!("<g{}>\n{}\n</g>", clip_attr, inner)
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- fmt_f64 ---

    #[test]
    fn fmt_f64_rounds_to_two_places() {
        assert_eq!(fmt_f64(50.123456789), "50.12");
        assert_eq!(fmt_f64(50.0), "50.00");
        assert_eq!(fmt_f64(0.005), "0.01"); // rounds up
        assert_eq!(fmt_f64(-3.14159), "-3.14");
    }

    // --- circle ---

    #[test]
    fn circle_contains_expected_attributes() {
        let svg = circle(50.0, 60.0, 10.0, "#FF0000", 1.0);
        assert!(svg.contains(r#"cx="50.00""#), "missing cx: {svg}");
        assert!(svg.contains(r#"cy="60.00""#), "missing cy: {svg}");
        assert!(svg.contains(r#"r="10.00""#), "missing r: {svg}");
        assert!(svg.contains("#FF0000"), "missing fill: {svg}");
        assert!(svg.contains(r#"opacity="1.00""#), "missing opacity: {svg}");
    }

    #[test]
    fn circle_coordinates_use_two_decimal_places() {
        let svg = circle(1.23456, 7.89012, 3.14159, "blue", 0.5);
        assert!(svg.contains("1.23"), "{svg}");
        assert!(svg.contains("7.89"), "{svg}");
        assert!(svg.contains("3.14"), "{svg}");
        assert!(svg.contains("0.50"), "{svg}");
        // Ensure the raw long decimal is NOT present
        assert!(!svg.contains("1.23456"), "{svg}");
    }

    // --- rect ---

    #[test]
    fn rect_with_zero_rx_omits_rx_attribute() {
        let svg = rect(0.0, 0.0, 100.0, 50.0, "#000", 0.0);
        assert!(!svg.contains("rx"), "rx should be absent: {svg}");
    }

    #[test]
    fn rect_with_nonzero_rx_includes_rx_attribute() {
        let svg = rect(0.0, 0.0, 100.0, 50.0, "#000", 4.0);
        assert!(svg.contains(r#"rx="4.00""#), "rx missing: {svg}");
    }

    // --- line ---

    #[test]
    fn line_contains_all_attributes() {
        let svg = line(0.0, 0.0, 100.0, 100.0, "grey", 1.5);
        assert!(svg.contains(r#"x1="0.00""#));
        assert!(svg.contains(r#"y1="0.00""#));
        assert!(svg.contains(r#"x2="100.00""#));
        assert!(svg.contains(r#"y2="100.00""#));
        assert!(svg.contains(r#"stroke="grey""#));
        assert!(svg.contains(r#"stroke-width="1.50""#));
    }

    // --- text ---

    #[test]
    fn text_with_zero_rotate_omits_transform() {
        let svg = text(10.0, 20.0, "Hello", "middle", 12, "#333", 0.0);
        assert!(!svg.contains("transform"), "transform should be absent: {svg}");
        assert!(svg.contains("Hello"), "{svg}");
    }

    #[test]
    fn text_with_nonzero_rotate_includes_transform() {
        let svg = text(10.0, 20.0, "Label", "end", 12, "#333", -90.0);
        assert!(svg.contains("transform"), "transform should be present: {svg}");
        assert!(svg.contains("rotate"), "{svg}");
    }

    #[test]
    fn text_escapes_xml_special_chars() {
        let svg = text(0.0, 0.0, "<hello & world>", "start", 12, "black", 0.0);
        assert!(svg.contains("&lt;hello &amp; world&gt;"), "{svg}");
    }

    // --- points_to_path ---

    #[test]
    fn points_to_path_empty_returns_empty_string() {
        assert_eq!(points_to_path(&[]), "");
    }

    #[test]
    fn points_to_path_two_points_correct_format() {
        let result = points_to_path(&[(1.0, 2.0), (3.0, 4.0)]);
        assert_eq!(result, "M 1.00,2.00 L 3.00,4.00");
    }

    #[test]
    fn points_to_path_single_point_is_just_move() {
        let result = points_to_path(&[(5.0, 10.0)]);
        assert_eq!(result, "M 5.00,10.00");
    }

    // --- polyline ---

    #[test]
    fn polyline_empty_input_returns_empty_string() {
        assert_eq!(polyline(&[], "black", 1.0, ""), "");
    }

    #[test]
    fn polyline_solid_line_omits_dasharray() {
        let svg = polyline(&[(0.0, 0.0), (10.0, 10.0)], "black", 1.0, "");
        assert!(!svg.contains("stroke-dasharray"), "{svg}");
        assert!(svg.contains(r#"fill="none""#), "{svg}");
    }

    #[test]
    fn polyline_dashed_line_includes_dasharray() {
        let svg = polyline(&[(0.0, 0.0), (10.0, 10.0)], "blue", 1.0, "6 3");
        assert!(svg.contains(r#"stroke-dasharray="6 3""#), "{svg}");
    }

    // --- polygon ---

    #[test]
    fn polygon_empty_input_returns_empty_string() {
        assert_eq!(polygon(&[], "red", "none", 1.0), "");
    }

    #[test]
    fn polygon_includes_fill_and_closes_path() {
        let svg = polygon(&[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)], "#aaa", "none", 0.6);
        assert!(svg.contains("Z"), "path not closed: {svg}");
        assert!(svg.contains("#aaa"), "{svg}");
        assert!(svg.contains("0.60"), "{svg}");
    }

    // --- group ---

    #[test]
    fn group_without_clip_has_no_clip_path_attr() {
        let elements = vec!["<circle/>".to_string()];
        let svg = group(&elements, None);
        assert!(!svg.contains("clip-path"), "{svg}");
        assert!(svg.starts_with("<g>"), "{svg}");
    }

    #[test]
    fn group_with_clip_includes_clip_path_attr() {
        let elements = vec!["<rect/>".to_string()];
        let svg = group(&elements, Some("plot_clip"));
        assert!(svg.contains(r#"clip-path="url(#plot_clip)""#), "{svg}");
    }
}

#[cfg(test)]
mod dashed_line_tests {
    use super::*;

    #[test]
    fn dashed_line_solid_omits_dasharray() {
        let svg = dashed_line(0.0, 0.0, 100.0, 100.0, "grey", 1.5, "");
        assert!(!svg.contains("stroke-dasharray"), "solid must omit stroke-dasharray: {svg}");
        assert!(svg.contains(r#"stroke="grey""#));
    }

    #[test]
    fn dashed_line_with_dash_includes_dasharray() {
        let svg = dashed_line(0.0, 50.0, 200.0, 50.0, "#aaa", 1.5, "6 3");
        assert!(svg.contains(r#"stroke-dasharray="6 3""#), "must include dasharray: {svg}");
    }

    #[test]
    fn dashed_line_coordinates_correct() {
        let svg = dashed_line(10.0, 20.0, 30.0, 40.0, "black", 2.0, "2 2");
        assert!(svg.contains(r#"x1="10.00""#));
        assert!(svg.contains(r#"y1="20.00""#));
        assert!(svg.contains(r#"x2="30.00""#));
        assert!(svg.contains(r#"y2="40.00""#));
    }
}