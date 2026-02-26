#![allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum Theme {
    Default,
    Dark,
    Minimal,
    Colorblind,
}

pub(crate) struct ThemeConfig {
    pub background: &'static str,
    pub foreground: &'static str,
    pub grid_color: &'static str,
    pub axis_color: &'static str,
    pub text_color: &'static str,
    pub font_family: &'static str,
    pub font_size_px: u32,
    pub palette: &'static [&'static str],
}

impl ThemeConfig {
    pub(crate) fn from(theme: &Theme) -> ThemeConfig {
        match theme {
            Theme::Default => DEFAULT_THEME,
            Theme::Dark => DARK_THEME,
            Theme::Minimal => MINIMAL_THEME,
            Theme::Colorblind => COLORBLIND_THEME,
        }
    }
}

// --- Theme constants ---

const DEFAULT_PALETTE: &[&str] = &[
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
];

const DEFAULT_THEME: ThemeConfig = ThemeConfig {
    background: "#FFFFFF",
    foreground: "#F8F8F8",
    grid_color: "#E0E0E0",
    axis_color: "#333333",
    text_color: "#333333",
    font_family: "Arial, sans-serif",
    font_size_px: 12,
    palette: DEFAULT_PALETTE,
};

const DARK_PALETTE: &[&str] = &[
    "#5B8DB8", "#E8955A", "#62B875", "#D45F5F", "#9482C0", "#A8906E", "#E89CD0", "#AAAAAA",
];

const DARK_THEME: ThemeConfig = ThemeConfig {
    background: "#1E1E1E",
    foreground: "#2A2A2A",
    grid_color: "#3A3A3A",
    axis_color: "#CCCCCC",
    text_color: "#CCCCCC",
    font_family: "Arial, sans-serif",
    font_size_px: 12,
    palette: DARK_PALETTE,
};

const MINIMAL_PALETTE: &[&str] = &[
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
];

const MINIMAL_THEME: ThemeConfig = ThemeConfig {
    background: "#FFFFFF",
    foreground: "#FFFFFF",
    grid_color: "#FFFFFF", // no visible gridlines
    axis_color: "#333333",
    text_color: "#333333",
    font_family: "Arial, sans-serif",
    font_size_px: 12,
    palette: MINIMAL_PALETTE,
};

const WONG_PALETTE: &[&str] = &[
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
];

const COLORBLIND_THEME: ThemeConfig = ThemeConfig {
    background: "#FFFFFF",
    foreground: "#F8F8F8",
    grid_color: "#E0E0E0",
    axis_color: "#333333",
    text_color: "#333333",
    font_family: "Arial, sans-serif",
    font_size_px: 12,
    palette: WONG_PALETTE,
};

#[derive(Debug, Clone)]
pub enum ColorScale {
    Viridis,
    Plasma,
    RdBu,
    Greyscale,
}

impl ColorScale {
    pub fn interpolate(&self, t: f64) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);
        let stops = self.stops();
        interpolate_stops(stops, t)
    }

    fn stops(&self) -> &'static [(f64, u8, u8, u8)] {
        match self {
            ColorScale::Viridis => VIRIDIS_STOPS,
            ColorScale::Plasma => PLASMA_STOPS,
            ColorScale::RdBu => RDBU_STOPS,
            ColorScale::Greyscale => GREYSCALE_STOPS,
        }
    }
}

fn interpolate_stops(stops: &[(f64, u8, u8, u8)], t: f64) -> (u8, u8, u8) {
    let mut lo = &stops[0];
    let mut hi = &stops[stops.len() - 1];

    for i in 0..stops.len() - 1 {
        if t >= stops[i].0 && t <= stops[i + 1].0 {
            lo = &stops[i];
            hi = &stops[i + 1];
            break;
        }
    }

    let range = hi.0 - lo.0;
    if range < 1e-10 {
        return (lo.1, lo.2, lo.3);
    }

    let factor = (t - lo.0) / range;
    let r = (lo.1 as f64 + factor * (hi.1 as f64 - lo.1 as f64)).round() as u8;
    let g = (lo.2 as f64 + factor * (hi.2 as f64 - lo.2 as f64)).round() as u8;
    let b = (lo.3 as f64 + factor * (hi.3 as f64 - lo.3 as f64)).round() as u8;

    (r, g, b)
}

const VIRIDIS_STOPS: &[(f64, u8, u8, u8)] = &[
    (0.00, 68, 1, 84),
    (0.25, 59, 82, 139),
    (0.50, 33, 145, 140),
    (0.75, 94, 201, 98),
    (1.00, 253, 231, 37),
];

const PLASMA_STOPS: &[(f64, u8, u8, u8)] = &[
    (0.00, 13, 8, 135),
    (0.25, 126, 3, 168),
    (0.50, 204, 71, 120),
    (0.75, 248, 149, 64),
    (1.00, 240, 249, 33),
];

const RDBU_STOPS: &[(f64, u8, u8, u8)] = &[
    (0.00, 178, 24, 43),   // strong red
    (0.25, 239, 138, 98),  // light red
    (0.50, 255, 255, 255), // white midpoint
    (0.75, 103, 169, 207), // light blue
    (1.00, 33, 102, 172),  // strong blue
];

const GREYSCALE_STOPS: &[(f64, u8, u8, u8)] = &[
    (0.00, 0, 0, 0),       // black
    (1.00, 255, 255, 255), // white
];

#[cfg(test)]
mod tests {
    use super::*;

    // --- ThemeConfig::from tests ---

    #[test]
    fn test_default_theme_background() {
        let config = ThemeConfig::from(&Theme::Default);
        assert_eq!(config.background, "#FFFFFF");
    }

    #[test]
    fn test_dark_theme_background() {
        let config = ThemeConfig::from(&Theme::Dark);
        assert_eq!(config.background, "#1E1E1E");
    }

    #[test]
    fn test_minimal_theme_background() {
        let config = ThemeConfig::from(&Theme::Minimal);
        assert_eq!(config.background, "#FFFFFF");
    }

    #[test]
    fn test_colorblind_theme_background() {
        let config = ThemeConfig::from(&Theme::Colorblind);
        assert_eq!(config.background, "#FFFFFF");
    }

    // --- Wong palette tests ---

    #[test]
    fn test_wong_palette_has_eight_entries() {
        let config = ThemeConfig::from(&Theme::Colorblind);
        assert_eq!(config.palette.len(), 8);
    }

    #[test]
    fn test_wong_palette_no_duplicates() {
        let config = ThemeConfig::from(&Theme::Colorblind);
        let mut seen = std::collections::HashSet::new();
        for color in config.palette {
            assert!(
                seen.insert(color),
                "duplicate color in Wong palette: {color}"
            );
        }
    }

    #[test]
    fn test_wong_palette_exact_values() {
        let config = ThemeConfig::from(&Theme::Colorblind);
        assert_eq!(config.palette[0], "#E69F00");
        assert_eq!(config.palette[1], "#56B4E9");
        assert_eq!(config.palette[2], "#009E73");
        assert_eq!(config.palette[3], "#F0E442");
        assert_eq!(config.palette[4], "#0072B2");
        assert_eq!(config.palette[5], "#D55E00");
        assert_eq!(config.palette[6], "#CC79A7");
        assert_eq!(config.palette[7], "#000000");
    }

    // --- ColorScale::interpolate tests ---

    #[test]
    fn test_viridis_min() {
        let (r, g, b) = ColorScale::Viridis.interpolate(0.0);
        assert_eq!((r, g, b), (68, 1, 84));
    }

    #[test]
    fn test_viridis_max() {
        let (r, g, b) = ColorScale::Viridis.interpolate(1.0);
        assert_eq!((r, g, b), (253, 231, 37));
    }

    #[test]
    fn test_viridis_midpoint_in_range() {
        let (r, g, b) = ColorScale::Viridis.interpolate(0.5);
        // midpoint should be around (33, 145, 140) — teal
        assert!(r < 100, "r should be low at midpoint");
        assert!(g > 100, "g should be high at midpoint");
        assert!(b > 100, "b should be high at midpoint");
    }

    #[test]
    fn test_plasma_min() {
        let (r, g, b) = ColorScale::Plasma.interpolate(0.0);
        assert_eq!((r, g, b), (13, 8, 135));
    }

    #[test]
    fn test_plasma_max() {
        let (r, g, b) = ColorScale::Plasma.interpolate(1.0);
        assert_eq!((r, g, b), (240, 249, 33));
    }

    #[test]
    fn test_rdbu_min() {
        let (r, g, b) = ColorScale::RdBu.interpolate(0.0);
        assert_eq!((r, g, b), (178, 24, 43));
    }

    #[test]
    fn test_rdbu_max() {
        let (r, g, b) = ColorScale::RdBu.interpolate(1.0);
        assert_eq!((r, g, b), (33, 102, 172));
    }

    #[test]
    fn test_rdbu_midpoint_is_white() {
        let (r, g, b) = ColorScale::RdBu.interpolate(0.5);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_greyscale_min() {
        let (r, g, b) = ColorScale::Greyscale.interpolate(0.0);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_greyscale_max() {
        let (r, g, b) = ColorScale::Greyscale.interpolate(1.0);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_greyscale_midpoint() {
        let (r, g, b) = ColorScale::Greyscale.interpolate(0.5);
        // should be mid-grey — all channels roughly equal and near 128
        assert!((120..=136).contains(&r));
        assert_eq!(r, g);
        assert_eq!(g, b);
    }

    #[test]
    fn test_interpolate_clamps_below_zero() {
        // should not panic, should return minimum color
        let (r, g, b) = ColorScale::Viridis.interpolate(-1.0);
        assert_eq!((r, g, b), (68, 1, 84));
    }

    #[test]
    fn test_interpolate_clamps_above_one() {
        // should not panic, should return maximum color
        let (r, g, b) = ColorScale::Viridis.interpolate(2.0);
        assert_eq!((r, g, b), (253, 231, 37));
    }
}
