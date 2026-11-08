package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	_ "image/jpeg"
	_ "image/png"
)

var (
	errUnsupported = errors.New("unsupported_format")
	errCorrupt     = errors.New("corrupt_image")
	errDimensions  = errors.New("bad_dimensions")
)

const (
	minSide   = 128
	maxSide   = 20000
	maxPixels = 40_000_000
)

// sanitizeImage identifies the format from magic bytes (never from the client-supplied
// name/Content-Type), strips every metadata segment WITHOUT touching pixel data (so the
// model sees exactly the original pixels), and checks dimensions to block decompression
// bombs. EXIF/XMP/comment/text chunks can carry patient identifiers or GPS.
func sanitizeImage(b []byte) (clean []byte, kind string, w, h int, err error) {
	switch {
	case len(b) >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF:
		kind, clean, err = "jpeg", nil, nil
		clean, err = scrubJPEG(b)
	case len(b) >= 8 && bytes.Equal(b[:8], []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}):
		kind = "png"
		clean, err = scrubPNG(b)
	default:
		return nil, "", 0, 0, errUnsupported
	}
	if err != nil {
		return nil, kind, 0, 0, err
	}
	cfg, _, derr := image.DecodeConfig(bytes.NewReader(clean))
	if derr != nil {
		return nil, kind, 0, 0, errCorrupt
	}
	if cfg.Width < minSide || cfg.Height < minSide || cfg.Width > maxSide || cfg.Height > maxSide ||
		int64(cfg.Width)*int64(cfg.Height) > maxPixels {
		return nil, kind, cfg.Width, cfg.Height, errDimensions
	}
	return clean, kind, cfg.Width, cfg.Height, nil
}

// scrubJPEG keeps the structural markers and drops APP1..APP13, APP15 (EXIF, XMP, ICC, ...),
// COM, and non-JFIF APP0. APP14 (Adobe) is kept because it affects colour decoding.
func scrubJPEG(b []byte) ([]byte, error) {
	out := make([]byte, 0, len(b))
	out = append(out, 0xFF, 0xD8)
	i := 2
	for {
		if i+4 > len(b) {
			return nil, errCorrupt
		}
		if b[i] != 0xFF {
			return nil, errCorrupt
		}
		for i < len(b) && b[i] == 0xFF { // fill bytes
			i++
		}
		if i >= len(b) {
			return nil, errCorrupt
		}
		m := b[i]
		i++
		if m == 0xD9 { // EOI before SOS: no image data
			return nil, errCorrupt
		}
		if m == 0x00 || m == 0x01 || (m >= 0xD0 && m <= 0xD8) { // stand-alone markers
			return nil, errCorrupt
		}
		if i+2 > len(b) {
			return nil, errCorrupt
		}
		n := int(binary.BigEndian.Uint16(b[i : i+2]))
		if n < 2 || i+n > len(b) {
			return nil, errCorrupt
		}
		seg := b[i : i+n]
		start := i - 2
		i += n
		if m == 0xDA { // SOS: copy header and everything after verbatim (entropy data + EOI)
			out = append(out, b[start:i]...)
			out = append(out, b[i:]...)
			return out, nil
		}
		keep := true
		switch {
		case m == 0xE0:
			keep = len(seg) >= 7 && bytes.Equal(seg[2:7], []byte("JFIF\x00"))
			if keep && len(seg) > 16 { // JFIF with embedded thumbnail -> drop
				keep = false
			}
		case m == 0xEE: // APP14 Adobe: keep
		case m >= 0xE1 && m <= 0xEF, m == 0xFE:
			keep = false
		}
		if keep {
			out = append(out, b[start:i]...)
		}
	}
}

// scrubPNG keeps critical chunks plus tRNS and drops all other ancillary chunks
// (tEXt/iTXt/zTXt/eXIf/tIME/iCCP/APNG...). CRCs are verified.
func scrubPNG(b []byte) ([]byte, error) {
	out := make([]byte, 0, len(b))
	out = append(out, b[:8]...)
	i, first, sawIDAT, sawEnd := 8, true, false, false
	for i < len(b) && !sawEnd {
		if i+12 > len(b) {
			return nil, errCorrupt
		}
		n := int(binary.BigEndian.Uint32(b[i : i+4]))
		if n < 0 || n > len(b) || i+12+n > len(b) {
			return nil, errCorrupt
		}
		typ := string(b[i+4 : i+8])
		if crc32.ChecksumIEEE(b[i+4:i+8+n]) != binary.BigEndian.Uint32(b[i+8+n:i+12+n]) {
			return nil, errCorrupt
		}
		if first && typ != "IHDR" {
			return nil, errCorrupt
		}
		first = false
		switch typ {
		case "IHDR", "PLTE", "tRNS":
			out = append(out, b[i:i+12+n]...)
		case "IDAT":
			sawIDAT = true
			out = append(out, b[i:i+12+n]...)
		case "IEND":
			sawEnd = true
			out = append(out, b[i:i+12+n]...)
		default: // everything else is dropped (critical unknown chunks => reject)
			if typ[0] >= 'A' && typ[0] <= 'Z' {
				return nil, errCorrupt
			}
		}
		i += 12 + n
	}
	if !sawIDAT || !sawEnd {
		return nil, errCorrupt
	}
	return out, nil
}
