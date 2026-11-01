//! Uncompressed RGB PNG. Used for the decoded preview bytes.

pub fn encode_rgb(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let expect = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(3);
    assert_eq!(rgb.len(), expect);
    let mut raw = Vec::with_capacity(expect + height as usize);
    for y in 0..height as usize {
        raw.push(0);
        let start = y * width as usize * 3;
        raw.extend_from_slice(&rgb[start..start + width as usize * 3]);
    }
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    png.extend(chunk(b"IHDR", &ihdr));
    png.extend(chunk(b"IDAT", &zlib_store(&raw)));
    png.extend(chunk(b"IEND", &[]));
    png
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + data.len());
    body.extend_from_slice(kind);
    body.extend_from_slice(data);
    let crc = crc32(&body);
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend(body);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut i = 0;
    while i < data.len() || data.is_empty() {
        let n = (data.len() - i).min(65535);
        let last = i + n >= data.len();
        out.push(if last { 1 } else { 0 });
        let nlen = !(n as u16);
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(&data[i..i + n]);
        i += n;
        if last {
            break;
        }
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Decode an 8-bit RGB/RGBA non-interlaced PNG whose IDAT uses stored zlib
/// blocks (the layout [`encode_rgb`] writes). Returns `None` for every other
/// layout so callers can fall back to an external decoder.
pub fn decode_rgb(png: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut rest = png.strip_prefix(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])?;
    let mut ihdr: Option<&[u8]> = None;
    let mut idat = Vec::new();
    while rest.len() >= 12 {
        let len = u32::from_be_bytes(rest[0..4].try_into().ok()?) as usize;
        let kind = &rest[4..8];
        let end = 8usize.checked_add(len)?.checked_add(4)?;
        if rest.len() < end {
            return None;
        }
        let data = &rest[8..8 + len];
        match kind {
            b"IHDR" => ihdr = Some(data),
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        rest = &rest[end..];
    }
    let ihdr = ihdr?;
    if ihdr.len() != 13 {
        return None;
    }
    let width = u32::from_be_bytes(ihdr[0..4].try_into().ok()?);
    let height = u32::from_be_bytes(ihdr[4..8].try_into().ok()?);
    let channels = match ihdr[9] {
        2 => 3usize,
        6 => 4,
        _ => return None,
    };
    if ihdr[8] != 8 || ihdr[10] != 0 || ihdr[11] != 0 || ihdr[12] != 0 || width == 0 || height == 0
    {
        return None;
    }
    let raw = zlib_unstore(&idat)?;
    let stride = width as usize * channels;
    if raw.len() != (stride + 1) * height as usize {
        return None;
    }
    let mut prev = vec![0u8; stride];
    let mut curr = vec![0u8; stride];
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height as usize {
        let filter = raw[y * (stride + 1)];
        let scan = &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        unfilter(filter, scan, &prev, &mut curr, channels)?;
        for x in 0..width as usize {
            rgb.extend_from_slice(&curr[x * channels..x * channels + 3]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    Some((rgb, width, height))
}

fn zlib_unstore(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 6 || data[0] & 0x0F != 8 {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 2usize;
    loop {
        let header = *data.get(i)?;
        let last = header & 1 == 1;
        if header >> 1 != 0 {
            return None;
        }
        let len = u16::from_le_bytes([*data.get(i + 1)?, *data.get(i + 2)?]) as usize;
        let nlen = u16::from_le_bytes([*data.get(i + 3)?, *data.get(i + 4)?]);
        if nlen != !(len as u16) {
            return None;
        }
        let start = i + 5;
        let end = start.checked_add(len)?;
        out.extend_from_slice(data.get(start..end)?);
        i = end;
        if last {
            break;
        }
    }
    Some(out)
}

fn unfilter(filter: u8, scan: &[u8], prev: &[u8], curr: &mut [u8], bpp: usize) -> Option<()> {
    match filter {
        0 => curr.copy_from_slice(scan),
        1 => {
            for x in 0..scan.len() {
                let left = if x >= bpp { curr[x - bpp] } else { 0 };
                curr[x] = scan[x].wrapping_add(left);
            }
        }
        2 => {
            for x in 0..scan.len() {
                curr[x] = scan[x].wrapping_add(prev[x]);
            }
        }
        3 => {
            for x in 0..scan.len() {
                let left = if x >= bpp { curr[x - bpp] } else { 0 };
                curr[x] = scan[x].wrapping_add(((left as u16 + prev[x] as u16) / 2) as u8);
            }
        }
        4 => {
            for x in 0..scan.len() {
                let left = if x >= bpp { curr[x - bpp] } else { 0 };
                let up_left = if x >= bpp { prev[x - bpp] } else { 0 };
                curr[x] = scan[x].wrapping_add(paeth(left, prev[x], up_left));
            }
        }
        _ => return None,
    }
    Some(())
}

fn paeth(left: u8, up: u8, up_left: u8) -> u8 {
    let base = left as i32 + up as i32 - up_left as i32;
    let dl = (base - left as i32).abs();
    let du = (base - up as i32).abs();
    let dul = (base - up_left as i32).abs();
    if dl <= du && dl <= dul {
        left
    } else if du <= dul {
        up
    } else {
        up_left
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_rgb, encode_rgb};

    #[test]
    fn decode_round_trips_the_encoder_layout() {
        let rgb: Vec<u8> = (0..64 * 32 * 3).map(|index| (index % 251) as u8).collect();
        let png = encode_rgb(64, 32, &rgb);
        let (decoded, width, height) = decode_rgb(&png).expect("decode");
        assert_eq!((width, height), (64, 32));
        assert_eq!(decoded, rgb);
    }

    #[test]
    fn decode_refuses_compressed_or_interlaced_layouts() {
        assert!(decode_rgb(b"not a png").is_none());
        let rgb = vec![7u8; 16 * 16 * 3];
        let mut png = encode_rgb(16, 16, &rgb);
        // Interlace flag lives at IHDR data offset 12: 8 signature + 8 header + 12.
        png[8 + 8 + 12] = 1;
        assert!(decode_rgb(&png).is_none());
    }
}
