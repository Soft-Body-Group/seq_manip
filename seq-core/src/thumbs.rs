//! Thumbnail / playback-frame generation: decode, color-manage, shrink,
//! cache on disk, hand the frontend a data URL (or raw JPEG bytes).
//!
//! EXR (scene-linear) goes through the `exr` crate and an ACES-style display
//! transform (`ColorOpts`: input space, view transform, exposure — the Hill
//! RRT/ODT fit rather than a full OCIO config). LDR formats go through the
//! `image` crate untouched; movies and DPX go through the vendored ffmpeg.

use base64::Engine;
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use serde::Deserialize;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Longest edge of a generated thumbnail, in pixels.
const MAX_DIM: u32 = 256;

/// What color space the scene-linear EXR pixels are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum InputSpace {
    /// Linear with Rec.709/sRGB primaries.
    #[default]
    LinearSrgb,
    /// ACEScg (AP1 primaries) — converted to linear sRGB before display.
    AcesCg,
}

/// Display rendering applied to scene-linear data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum ViewTransform {
    /// ACES 1.x RRT+ODT (Stephen Hill's matrix fit) — the filmic look.
    #[default]
    Aces,
    /// Straight clamp + sRGB encode (highlights clip).
    Gamma22,
    /// No tone mapping, no encode gamma — raw values clamped.
    Raw,
}

/// Color decisions applied when decoding scene-linear sources (EXR). LDR
/// formats and ffmpeg-decoded media pass through untouched.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorOpts {
    pub input: InputSpace,
    pub view: ViewTransform,
    /// Exposure in stops, applied before the view transform.
    pub exposure: f32,
}

impl ColorOpts {
    /// Stable cache tag; exposure quantised to half-stops (the UI step).
    fn cache_tag(&self) -> (u8, u8, i32) {
        (
            self.input as u8,
            self.view as u8,
            (self.exposure * 2.0).round() as i32,
        )
    }
}

// ACEScg (AP1, D60) -> linear sRGB (Rec.709, D65), standard CTL-derived matrix.
const AP1_TO_SRGB: [[f32; 3]; 3] = [
    [1.70505, -0.62179, -0.08326],
    [-0.13026, 1.14080, -0.01055],
    [-0.02400, -0.12897, 1.15297],
];

// Stephen Hill's ACES RRT+ODT fit (BakingLab): operates on linear sRGB.
const ACES_INPUT: [[f32; 3]; 3] = [
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
];
const ACES_OUTPUT: [[f32; 3]; 3] = [
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
];

fn mat_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn rrt_odt_fit(v: f32) -> f32 {
    (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.432951) + 0.238081)
}

/// Real sRGB OETF (not a plain 1/2.2 power).
fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Pre-baked per-frame tone mapper: the exposure gain and the sRGB encode
/// curve are computed once (the curve as a LUT), so the per-pixel hot loop
/// over a multi-megapixel EXR has no `powf` calls — only multiplies, a small
/// matrix, and table lookups.
struct ToneMapper {
    gain: f32,
    input: InputSpace,
    view: ViewTransform,
    /// sRGB OETF sampled over [0, 1] -> 8-bit.
    srgb_lut: [u8; Self::LUT_N],
}

impl ToneMapper {
    const LUT_N: usize = 4096;

    fn new(opts: &ColorOpts) -> Self {
        let mut srgb_lut = [0u8; Self::LUT_N];
        for (i, slot) in srgb_lut.iter_mut().enumerate() {
            let v = i as f32 / (Self::LUT_N - 1) as f32;
            *slot = (srgb_encode(v) * 255.0 + 0.5) as u8;
        }
        Self {
            gain: 2f32.powf(opts.exposure),
            input: opts.input,
            view: opts.view,
            srgb_lut,
        }
    }

    #[inline]
    fn enc(&self, v: f32) -> u8 {
        let i = (v.clamp(0.0, 1.0) * (Self::LUT_N - 1) as f32) as usize;
        self.srgb_lut[i]
    }

    #[inline]
    fn map(&self, rgb: [f32; 3]) -> [u8; 3] {
        let mut v = [rgb[0] * self.gain, rgb[1] * self.gain, rgb[2] * self.gain];
        if self.input == InputSpace::AcesCg {
            v = mat_mul(&AP1_TO_SRGB, v);
        }
        match self.view {
            ViewTransform::Aces => {
                let fitted = mat_mul(&ACES_INPUT, v).map(rrt_odt_fit);
                let out = mat_mul(&ACES_OUTPUT, fitted);
                [self.enc(out[0]), self.enc(out[1]), self.enc(out[2])]
            }
            ViewTransform::Gamma22 => [self.enc(v[0]), self.enc(v[1]), self.enc(v[2])],
            ViewTransform::Raw => {
                v.map(|c| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            }
        }
    }
}

/// Formats decoded via the vendored ffmpeg sidecar: movie containers plus
/// DPX, which neither the `image` nor `exr` crate reads.
fn is_ffmpeg_format(path: &Path) -> bool {
    matches!(
        ext_of(path).as_deref(),
        Some("mov" | "mp4" | "m4v" | "avi" | "mkv" | "webm" | "mxf" | "dpx")
    )
}

fn ext_of(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
}

/// Extensions we can decode. Everything else returns `None` early so the
/// frontend doesn't even wait on a doomed decode.
pub fn displayable(path: &Path) -> bool {
    matches!(
        ext_of(path).as_deref(),
        Some("exr" | "png" | "jpg" | "jpeg" | "tif" | "tiff" | "gif" | "bmp" | "webp")
    ) || is_ffmpeg_format(path)
}

/// A `Command` for a subprocess (ffmpeg) that never shows a console window.
/// On Windows, spawning a console app from a GUI app pops a terminal per
/// process unless CREATE_NO_WINDOW is set — a folder of movies produced
/// hundreds of them. Every ffmpeg spawn in the app must go through this.
pub(crate) fn hidden_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    #[allow(unused_mut)]
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Locates the ffmpeg sidecar: next to the app binary in a bundle (Tauri
/// strips the target-triple suffix there), the triple-suffixed dev copy in
/// the target dir or `binaries/`, then PATH. Nothing is embedded: with no
/// ffmpeg found, thumbnails fall back to a glyph.
pub(crate) fn ffmpeg_path() -> Option<PathBuf> {
    let exe = std::env::consts::EXE_SUFFIX; // ".exe" on Windows, "" elsewhere
    let triple_name = format!(
        "ffmpeg-{}-{}{}",
        std::env::consts::ARCH,
        match std::env::consts::OS {
            "macos" => "apple-darwin",
            "linux" => "unknown-linux-gnu",
            "windows" => "pc-windows-msvc",
            other => other,
        },
        exe,
    );
    let plain = format!("ffmpeg{exe}");
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            for name in [plain.as_str(), triple_name.as_str()] {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    let dev = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("binaries")
        .join(&triple_name);
    if dev.is_file() {
        return Some(dev);
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(&plain))
            .find(|p| p.is_file())
    })
}

/// Extracts one frame via ffmpeg straight into `dst` (a PNG in the cache),
/// scaled to fit. Movies seek 1s in to skip slates/black leaders, retrying
/// at frame 0 for clips shorter than that. Returns false when ffmpeg is
/// unavailable or the decode fails — callers fall back to a glyph.
fn ffmpeg_thumb(src: &Path, dst: &Path) -> bool {
    let Some(ffmpeg) = ffmpeg_path() else {
        return false;
    };
    let is_movie = !matches!(ext_of(src).as_deref(), Some("dpx"));
    let scale = format!("scale='min({MAX_DIM},iw)':-2");

    let mut attempts: Vec<Vec<&str>> = Vec::new();
    if is_movie {
        attempts.push(vec!["-ss", "1"]);
    }
    attempts.push(Vec::new());

    for seek in attempts {
        let status = hidden_command(&ffmpeg)
            .args(["-y", "-loglevel", "error"])
            .args(&seek)
            .arg("-i")
            .arg(src)
            .args(["-frames:v", "1", "-vf", &scale])
            .arg(dst)
            .status();
        if status.map(|s| s.success()).unwrap_or(false)
            && fs::metadata(dst).map(|m| m.len() > 0).unwrap_or(false)
        {
            return true;
        }
    }
    false
}

/// Returns a `data:image/png;base64,…` URL for a downscaled preview of
/// `path`, using (and filling) an on-disk cache keyed by path + mtime + size
/// (+ color decisions where they apply).
pub fn thumbnail_data_url(
    path: &Path,
    cache_dir: &Path,
    color: &ColorOpts,
) -> Result<Option<String>, String> {
    if !displayable(path) {
        return Ok(None);
    }
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    meta.len().hash(&mut hasher);
    if let Ok(modified) = meta.modified() {
        modified.hash(&mut hasher);
    }
    // Color only affects natively-decoded scene-linear sources.
    if !is_ffmpeg_format(path) {
        color.cache_tag().hash(&mut hasher);
    }
    let cache_path = cache_dir.join(format!("{:016x}.png", hasher.finish()));

    let png = match fs::read(&cache_path) {
        Ok(bytes) => bytes,
        Err(_) if is_ffmpeg_format(path) => {
            // ffmpeg writes the scaled PNG straight into the cache slot.
            let _ = fs::create_dir_all(cache_dir);
            if !ffmpeg_thumb(path, &cache_path) {
                return Ok(None);
            }
            fs::read(&cache_path).map_err(|e| e.to_string())?
        }
        Err(_) => {
            let rgba = decode(path, color)?;
            let scaled = shrink(rgba);
            let mut bytes = Vec::new();
            DynamicImage::ImageRgba8(scaled)
                .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            // Cache failures are not fatal; the thumb still renders.
            let _ = fs::create_dir_all(cache_dir);
            let _ = fs::write(&cache_path, &bytes);
            bytes
        }
    };

    Ok(Some(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )))
}

fn decode(path: &Path, color: &ColorOpts) -> Result<RgbaImage, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "exr" {
        decode_exr(path, color)
    } else {
        // LDR formats are already display-referred; no transform.
        Ok(image::open(path).map_err(|e| e.to_string())?.to_rgba8())
    }
}

/// Reads the first RGBA layer of an EXR (scene-linear) and renders it to
/// display 8-bit through the chosen color transform.
fn decode_exr(path: &Path, color: &ColorOpts) -> Result<RgbaImage, String> {
    let tm = ToneMapper::new(color);
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| {
            RgbaImage::new(resolution.width() as u32, resolution.height() as u32)
        },
        move |img, pos, (r, g, b, a): (f32, f32, f32, f32)| {
            let [r8, g8, b8] = tm.map([r.max(0.0), g.max(0.0), b.max(0.0)]);
            let alpha = (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            img.put_pixel(pos.x() as u32, pos.y() as u32, Rgba([r8, g8, b8, alpha]));
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(image.layer_data.channel_data.pixels)
}

fn shrink(img: RgbaImage) -> RgbaImage {
    shrink_to(img, MAX_DIM)
}

fn shrink_to(img: RgbaImage, max_dim: u32) -> RgbaImage {
    let (w, h) = (img.width(), img.height());
    if w <= max_dim && h <= max_dim {
        return img;
    }
    let scale = max_dim as f32 / w.max(h) as f32;
    let (nw, nh) = (
        ((w as f32 * scale) as u32).max(1),
        ((h as f32 * scale) as u32).max(1),
    );
    image::imageops::thumbnail(&img, nw, nh)
}

/// One playback frame: decoded (native or via ffmpeg), scaled to fit
/// `max_dim`, JPEG-encoded, and disk-cached so a sequence transcodes once
/// per resolution. Returns the JPEG bytes for the frontend's RAM buffer.
pub fn playback_jpeg(
    path: &Path,
    cache_dir: &Path,
    max_dim: u32,
    color: &ColorOpts,
) -> Result<Vec<u8>, String> {
    let ext = ext_of(path);
    let is_exr = ext.as_deref() == Some("exr");
    let native_still = !is_ffmpeg_format(path) && !is_exr;

    // Fast path: a browser-decodable still already within the target size —
    // serve the original bytes untouched (no decode → resize → re-encode).
    // The browser decodes JPEG/PNG far faster than a Rust round-trip, and the
    // GPU canvas downsamples for display.
    if native_still {
        if let Ok((w, h)) = image::image_dimensions(path) {
            if w.max(h) <= max_dim {
                return fs::read(path).map_err(|e| e.to_string());
            }
        }
    }

    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    meta.len().hash(&mut hasher);
    if let Ok(modified) = meta.modified() {
        modified.hash(&mut hasher);
    }
    ("play", max_dim).hash(&mut hasher);
    if !is_ffmpeg_format(path) {
        color.cache_tag().hash(&mut hasher);
    }
    let cache_path = cache_dir.join(format!("{:016x}.jpg", hasher.finish()));

    if let Ok(bytes) = fs::read(&cache_path) {
        return Ok(bytes);
    }
    let _ = fs::create_dir_all(cache_dir);

    if is_ffmpeg_format(path) {
        if !ffmpeg_extract(path, &cache_path, max_dim) {
            return Err(format!("cannot decode {}", path.display()));
        }
        return fs::read(&cache_path).map_err(|e| e.to_string());
    }

    let rgb = image::DynamicImage::ImageRgba8(shrink_to(decode(path, color)?, max_dim)).to_rgb8();
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut Cursor::new(&mut bytes), 85)
        .encode_image(&rgb)
        .map_err(|e| e.to_string())?;
    let _ = fs::write(&cache_path, &bytes);
    Ok(bytes)
}

/// ffmpeg one-frame extraction into `dst` (format from the extension),
/// scaled to fit `max_dim`.
fn ffmpeg_extract(src: &Path, dst: &Path, max_dim: u32) -> bool {
    let Some(ffmpeg) = ffmpeg_path() else {
        return false;
    };
    let scale = format!("scale='min({max_dim},iw)':-2");
    hidden_command(&ffmpeg)
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(src)
        .args(["-frames:v", "1", "-vf", &scale])
        .arg(dst)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
        && fs::metadata(dst).map(|m| m.len() > 0).unwrap_or(false)
}

/// Picks the representative frame for an entity preview: the middle existing
/// frame of a sequence reads better than the first (slates, black leaders).
pub fn representative_path(items: &[PathBuf]) -> Option<PathBuf> {
    items.get(items.len() / 2).cloned()
}

/// Whether this is a movie container we can time-sample for scrubbing.
pub fn is_movie(path: &Path) -> bool {
    matches!(
        ext_of(path).as_deref(),
        Some("mov" | "mp4" | "m4v" | "avi" | "mkv" | "webm" | "mxf")
    )
}

/// Movie duration in seconds, parsed from ffmpeg's own banner (avoids
/// vendoring ffprobe too). `Duration: HH:MM:SS.cc` on stderr.
fn movie_duration(ffmpeg: &Path, path: &Path) -> Option<f64> {
    let output = hidden_command(ffmpeg)
        .arg("-i")
        .arg(path)
        .output()
        .ok()?;
    let banner = String::from_utf8_lossy(&output.stderr);
    let stamp = banner.split("Duration: ").nth(1)?.get(..11)?;
    let mut parts = stamp.split(':');
    let h: f64 = parts.next()?.trim().parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let s: f64 = parts.next()?.parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

/// Evenly time-sampled scrub frames for a movie, as data URLs. Each sample
/// gets its own cache slot (path + mtime + sample index), so a movie only
/// decodes once. Failed samples come back as empty strings.
pub fn movie_scrub_data_urls(path: &Path, cache_dir: &Path, count: usize) -> Vec<String> {
    let Some(ffmpeg) = ffmpeg_path() else {
        return Vec::new();
    };
    let Some(duration) = movie_duration(&ffmpeg, path) else {
        return Vec::new();
    };
    let Ok(meta) = fs::metadata(path) else {
        return Vec::new();
    };
    let n = count.clamp(2, 24);
    let scale = format!("scale='min({MAX_DIM},iw)':-2");
    let _ = fs::create_dir_all(cache_dir);

    (0..n)
        .map(|i| {
            // Stop at 96% so the last seek lands before EOF.
            let t = duration * 0.96 * i as f64 / (n - 1) as f64;

            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);
            meta.len().hash(&mut hasher);
            if let Ok(modified) = meta.modified() {
                modified.hash(&mut hasher);
            }
            ("scrub", i).hash(&mut hasher);
            let cache_path = cache_dir.join(format!("{:016x}.png", hasher.finish()));

            let png = match fs::read(&cache_path) {
                Ok(bytes) => bytes,
                Err(_) => {
                    let ok = hidden_command(&ffmpeg)
                        .args(["-y", "-loglevel", "error", "-ss"])
                        .arg(format!("{t:.3}"))
                        .arg("-i")
                        .arg(path)
                        .args(["-frames:v", "1", "-vf", &scale])
                        .arg(&cache_path)
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false);
                    if !ok {
                        return String::new();
                    }
                    fs::read(&cache_path).unwrap_or_default()
                }
            };
            if png.is_empty() {
                return String::new();
            }
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(png)
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thumbnails_a_png_and_caches_it() {
        let dir = std::env::temp_dir().join(format!(
            "seq_manip_thumbs_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let img_path = dir.join("test.png");
        RgbaImage::from_pixel(640, 480, Rgba([200, 30, 30, 255]))
            .save(&img_path)
            .unwrap();

        let cache = dir.join("cache");
        let url = thumbnail_data_url(&img_path, &cache, &ColorOpts::default()).unwrap().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        // Cached on disk, and a second call serves the same bytes.
        assert_eq!(fs::read_dir(&cache).unwrap().count(), 1);
        assert_eq!(thumbnail_data_url(&img_path, &cache, &ColorOpts::default()).unwrap().unwrap(), url);

        // Unsupported extension short-circuits.
        let dpx = dir.join("frame.dpx");
        fs::write(&dpx, b"not an image").unwrap();
        assert!(thumbnail_data_url(&dpx, &cache, &ColorOpts::default()).unwrap().is_none());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn thumbnails_mov_and_dpx_via_ffmpeg() {
        let Some(ffmpeg) = ffmpeg_path() else {
            // Vendored binary absent (fresh clone before fetch_ffmpeg.sh);
            // nothing to test.
            return;
        };
        let dir = std::env::temp_dir().join(format!(
            "seq_manip_ffm_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("frame.png");
        RgbaImage::from_pixel(320, 200, Rgba([40, 120, 200, 255]))
            .save(&src)
            .unwrap();

        // Author a dpx and a (single-frame) mov from the png.
        let dpx = dir.join("frame.dpx");
        let mov = dir.join("clip.mov");
        for out in [&dpx, &mov] {
            let ok = hidden_command(&ffmpeg)
                .args(["-y", "-loglevel", "error", "-i"])
                .arg(&src)
                .arg(out)
                .status()
                .unwrap()
                .success();
            assert!(ok, "ffmpeg failed to author {}", out.display());
        }

        let cache = dir.join("cache");
        for media in [&dpx, &mov] {
            let url = thumbnail_data_url(media, &cache, &ColorOpts::default()).unwrap();
            assert!(
                url.is_some_and(|u| u.starts_with("data:image/png;base64,")),
                "no thumb for {}",
                media.display()
            );
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn thumbnails_an_exr_round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "seq_manip_exr_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let exr_path = dir.join("frame.exr");
        exr::prelude::write_rgba_file(&exr_path, 320, 200, |x, _y| {
            // HDR values above 1.0 exercise the tone-map.
            (x as f32 / 80.0, 0.25, 0.1, 1.0)
        })
        .unwrap();

        let cache = dir.join("cache");
        let url = thumbnail_data_url(&exr_path, &cache, &ColorOpts::default()).unwrap().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));

        fs::remove_dir_all(&dir).unwrap();
    }
}
