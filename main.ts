// png-squeeze-01-run-20261127 entry: a lossless PNG re-encoder that squeezes
// 1280x720 RGBA screenshots into as few bytes as it can.
//
// What it does, per screenshot:
//   1. Decodes the stored input PNG to raw RGBA.
//   2. Analyses the pixels: whether every pixel is opaque, whether every pixel
//      is grey, and whether the image holds at most 256 distinct colours.
//   3. Tries the lossless forms that can hold it exactly -- a palette form
//      (colour type 3, in count order and in luminance order), and the channel
//      form (grey / grey+alpha / RGB / RGBA, with a tRNS key when one fully
//      transparent colour can stand for every transparent pixel) -- and, for
//      each form, the per-row filter choice that compresses best.
//   4. Writes the winner as a single-zlib-stream PNG holding only the chunks
//      the contest reads (IHDR, PLTE, tRNS, IDAT, IEND).
//
// Determinism: every choice is a function of the input bytes alone. No wall
// clock is read, no environment value varies the search, and every search is
// cut by a fixed amount of work, so two runs write byte-identical files.
//
// Standard library only (node:fs, node:path, node:zlib). Node 22, erasable
// TypeScript: type annotations only, no enums or namespaces.

import { inflateSync, deflateSync, constants } from 'node:zlib';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ---------------------------------------------------------------------------
// CRC-32 (ISO 3309, the one PNG chunks carry).
// ---------------------------------------------------------------------------

const CRC_TABLE: Int32Array = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(bytes: Uint8Array, start: number, end: number): number {
  let c = -1;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

// ---------------------------------------------------------------------------
// PNG writing: the signature, then each chunk with length, type and CRC.
// ---------------------------------------------------------------------------

const SIGNATURE = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);

function pngOf(chunks: Array<[string, Uint8Array]>): Uint8Array {
  let size = SIGNATURE.length;
  for (const [, data] of chunks) size += 12 + data.length;
  const out = new Uint8Array(size);
  const view = new DataView(out.buffer);
  out.set(SIGNATURE, 0);
  let at = SIGNATURE.length;
  for (const [type, data] of chunks) {
    view.setUint32(at, data.length);
    for (let i = 0; i < 4; i++) out[at + 4 + i] = type.charCodeAt(i);
    out.set(data, at + 8);
    view.setUint32(at + 8 + data.length, crc32(out, at + 4, at + 8 + data.length));
    at += 12 + data.length;
  }
  return out;
}

function ihdrOf(width: number, height: number, colourType: number): Uint8Array {
  const d = new Uint8Array(13);
  const v = new DataView(d.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  d[8] = 8; // bit depth 8
  d[9] = colourType;
  d[10] = 0; // compression method 0
  d[11] = 0; // filter method 0
  d[12] = 0; // no interlace
  return d;
}

// ---------------------------------------------------------------------------
// The five PNG row filters, written from the PNG specification (section 12.8).
// ---------------------------------------------------------------------------

function paethPredictor(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = p >= a ? p - a : a - p;
  const pb = p >= b ? p - b : b - p;
  const pc = p >= c ? p - c : c - p;
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

// Filter row `cur` (stride samples, bpp bytes a pixel; `prev` the row above,
// null on the first row) by `type` into `out`. All arithmetic is mod 256, as
// Uint8Array assignment keeps it.
function filterRow(type: number, cur: Uint8Array, prev: Uint8Array | null, bpp: number, out: Uint8Array): void {
  const n = cur.length;
  if (type === 0) {
    out.set(cur);
  } else if (type === 1) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (i >= bpp ? cur[i - bpp] : 0);
  } else if (type === 2) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (prev === null ? 0 : prev[i]);
  } else if (type === 3) {
    for (let i = 0; i < n; i++) {
      const left = i >= bpp ? cur[i - bpp] : 0;
      const up = prev === null ? 0 : prev[i];
      out[i] = cur[i] - ((left + up) >> 1);
    }
  } else {
    for (let i = 0; i < n; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev === null ? 0 : prev[i];
      const c = prev !== null && i >= bpp ? prev[i - bpp] : 0;
      out[i] = cur[i] - paethPredictor(a, b, c);
    }
  }
}

// The sum of a filtered row's bytes read as signed values, absolute: the
// minimum-sum heuristic's cost (PNG specification section 12.8).
function signedSum(row: Uint8Array): number {
  let s = 0;
  for (let i = 0; i < row.length; i++) {
    const v = row[i];
    s += v < 128 ? v : 256 - v;
  }
  return s;
}

// ---------------------------------------------------------------------------
// PNG reading: the stored input PNGs are colour type 6, unfiltered, one zlib
// stream in stored blocks, but the decoder below reads every colour type the
// contest accepts (0, 2, 3, 4, 6), bit depth 8, not interlaced, so it holds
// for any input the harness hands it.
// ---------------------------------------------------------------------------

const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

interface DecodedImage {
  width: number;
  height: number;
  rgba: Uint8Array;
}

function decodePng(bytes: Uint8Array): DecodedImage {
  if (bytes.length < 8) throw new Error('input too short for a PNG signature');
  for (let i = 0; i < 8; i++) {
    if (bytes[i] !== SIGNATURE[i]) throw new Error('input is not a PNG');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 8;
  let header: { width: number; height: number; colourType: number } | null = null;
  let palette: Uint8Array | null = null;
  let trns: Uint8Array | null = null;
  const idatParts: Uint8Array[] = [];
  let ended = false;
  while (at < bytes.length) {
    if (ended) throw new Error('bytes after IEND');
    if (at + 12 > bytes.length) throw new Error('truncated chunk header');
    const len = view.getUint32(at);
    if (at + 12 + len > bytes.length) throw new Error('truncated chunk');
    const type = String.fromCharCode(bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]);
    if (crc32(bytes, at + 4, at + 8 + len) !== view.getUint32(at + 8 + len)) {
      throw new Error(`chunk ${type} fails its CRC`);
    }
    const data = bytes.subarray(at + 8, at + 8 + len);
    at += 12 + len;
    if (type === 'IHDR') {
      if (header !== null) throw new Error('two IHDR chunks');
      const dv = new DataView(data.buffer, data.byteOffset, 13);
      header = { width: dv.getUint32(0), height: dv.getUint32(4), colourType: data[9] };
      if (data[8] !== 8) throw new Error('bit depth is not 8');
      if (data[12] !== 0) throw new Error('interlaced input');
      if (!(header.colourType in CHANNELS)) throw new Error('unsupported colour type');
    } else if (type === 'PLTE') {
      palette = data;
    } else if (type === 'tRNS') {
      trns = data;
    } else if (type === 'IDAT') {
      idatParts.push(data);
    } else if (type === 'IEND') {
      ended = true;
    } else {
      throw new Error(`unexpected chunk ${type}`);
    }
  }
  if (!ended || header === null || idatParts.length === 0) throw new Error('incomplete PNG');
  const { width, height, colourType } = header;
  const bpp = CHANNELS[colourType];
  const stride = width * bpp;
  const expected = height * (stride + 1);
  const idat = idatParts.length === 1 ? idatParts[0] : Buffer.concat(idatParts);
  const raw = inflateSync(idat, { maxOutputLength: expected + 1 });
  if (raw.length !== expected) throw new Error('IDAT inflates to the wrong size');

  // Unfilter in place, row by row.
  const samples = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const ft = raw[y * (stride + 1)];
    if (ft > 4) throw new Error(`row ${y} has filter type ${ft}`);
    const src = y * (stride + 1) + 1;
    const o = y * stride;
    const p = o - stride;
    if (ft === 0) {
      samples.set(raw.subarray(src, src + stride), o);
    } else if (ft === 1) {
      for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (i >= bpp ? samples[o + i - bpp] : 0);
    } else if (ft === 2) {
      for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (y > 0 ? samples[p + i] : 0);
    } else if (ft === 3) {
      for (let i = 0; i < stride; i++) {
        const left = i >= bpp ? samples[o + i - bpp] : 0;
        const up = y > 0 ? samples[p + i] : 0;
        samples[o + i] = raw[src + i] + ((left + up) >> 1);
      }
    } else {
      for (let i = 0; i < stride; i++) {
        const a = i >= bpp ? samples[o + i - bpp] : 0;
        const b = y > 0 ? samples[p + i] : 0;
        const c = y > 0 && i >= bpp ? samples[p + i - bpp] : 0;
        samples[o + i] = raw[src + i] + paethPredictor(a, b, c);
      }
    }
  }

  // Expand to RGBA.
  const n = width * height;
  const rgba = new Uint8Array(n * 4);
  if (colourType === 6) {
    rgba.set(samples);
  } else if (colourType === 2) {
    const key = trns === null ? -1 : (trns[1] << 16) | (trns[3] << 8) | trns[5];
    for (let i = 0, s = 0, d = 0; i < n; i++, s += 3, d += 4) {
      rgba[d] = samples[s];
      rgba[d + 1] = samples[s + 1];
      rgba[d + 2] = samples[s + 2];
      rgba[d + 3] = ((samples[s] << 16) | (samples[s + 1] << 8) | samples[s + 2]) === key ? 0 : 255;
    }
  } else if (colourType === 0) {
    const key = trns === null ? -1 : trns[1];
    for (let i = 0, d = 0; i < n; i++, d += 4) {
      const g = samples[i];
      rgba[d] = g;
      rgba[d + 1] = g;
      rgba[d + 2] = g;
      rgba[d + 3] = g === key ? 0 : 255;
    }
  } else if (colourType === 4) {
    for (let i = 0, s = 0, d = 0; i < n; i++, s += 2, d += 4) {
      rgba[d] = samples[s];
      rgba[d + 1] = samples[s];
      rgba[d + 2] = samples[s];
      rgba[d + 3] = samples[s + 1];
    }
  } else {
    if (palette === null) throw new Error('palette image without PLTE');
    const entries = palette.length / 3;
    for (let i = 0, d = 0; i < n; i++, d += 4) {
      const e = samples[i];
      if (e >= entries) throw new Error('palette index past the palette');
      rgba[d] = palette[e * 3];
      rgba[d + 1] = palette[e * 3 + 1];
      rgba[d + 2] = palette[e * 3 + 2];
      rgba[d + 3] = trns !== null && e < trns.length ? trns[e] : 255;
    }
  }
  return { width, height, rgba };
}

// ---------------------------------------------------------------------------
// Analysis: what the image holds, so the encoder can pick the smallest form
// that still holds it exactly.
// ---------------------------------------------------------------------------

interface Colour {
  word: number;
  r: number;
  g: number;
  b: number;
  a: number;
  count: number;
}

interface Analysis {
  opaque: boolean;
  grey: boolean;
  colours: Colour[] | null; // null when more than 256 distinct colours
}

function analyse(rgba: Uint8Array): Analysis {
  const n = rgba.length >>> 2;
  let opaque = true;
  let grey = true;
  const counts = new Map<number, number>();
  let many = false;
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    if (rgba[d + 3] !== 255) opaque = false;
    if (rgba[d] !== rgba[d + 1] || rgba[d] !== rgba[d + 2]) grey = false;
    if (!many) {
      const word = (rgba[d] << 24) | (rgba[d + 1] << 16) | (rgba[d + 2] << 8) | rgba[d + 3];
      const k = counts.get(word);
      if (k === undefined) {
        if (counts.size === 256) many = true;
        else counts.set(word, 1);
      } else {
        counts.set(word, k + 1);
      }
    }
  }
  if (many) return { opaque, grey, colours: null };
  const colours: Colour[] = [];
  for (const [word, count] of counts) {
    colours.push({
      word,
      r: (word >>> 24) & 0xff,
      g: (word >>> 16) & 0xff,
      b: (word >>> 8) & 0xff,
      a: word & 0xff,
      count,
    });
  }
  return { opaque, grey, colours };
}

// The one fully transparent value (grey sample or RGB triple) the image may
// key by tRNS instead of carrying an alpha channel: every non-opaque pixel
// is that exact value at alpha 0, and no opaque pixel holds that value.
// Null when there is no such key.
function transparentKey(rgba: Uint8Array, grey: boolean): number {
  let key = -1;
  for (let d = 0; d < rgba.length; d += 4) {
    const a = rgba[d + 3];
    if (a === 255) continue;
    if (a !== 0) return -1;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (key === -1) key = v;
    else if (key !== v) return -1;
  }
  if (key === -1) return -1;
  for (let d = 0; d < rgba.length; d += 4) {
    if (rgba[d + 3] !== 255) continue;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (v === key) return -1;
  }
  return key;
}

// ---------------------------------------------------------------------------
// Forms: the lossless shapes an image can be written in.
// ---------------------------------------------------------------------------

interface Form {
  colourType: number;
  bpp: number;
  samples: Uint8Array;
  plte: Uint8Array | null;
  trns: Uint8Array | null;
}


// A palette form. Transparent entries come first (by alpha, so tRNS is
// short), then the rest by `order`: most frequent first ('count'), or by
// luminance ('luminance'), which groups nearby indices and filters better
// on smooth pictures.
function paletteForm(rgba: Uint8Array, colours: Colour[], order: 'count' | 'luminance'): Form {
  const lum = (c: Colour): number => 299 * c.r + 587 * c.g + 114 * c.b;
  const entries = [...colours].sort((x, y) => {
    const ta = x.a === 255 ? 1 : 0;
    const tb = y.a === 255 ? 1 : 0;
    if (ta !== tb) return ta - tb;
    if (ta === 0 && x.a !== y.a) return x.a - y.a;
    if (order === 'luminance') return lum(x) - lum(y) || x.word - y.word;
    return y.count - x.count || x.word - y.word;
  });
  const index = new Map<number, number>();
  entries.forEach((c, i) => index.set(c.word, i));
  const plte = new Uint8Array(entries.length * 3);
  let transparent = 0;
  entries.forEach((c, i) => {
    plte[i * 3] = c.r;
    plte[i * 3 + 1] = c.g;
    plte[i * 3 + 2] = c.b;
    if (c.a !== 255) transparent = i + 1;
  });
  const trns = transparent === 0 ? null : Uint8Array.from(entries.slice(0, transparent), (c) => c.a);
  const n = rgba.length >>> 2;
  const indices = new Uint8Array(n);
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    const word = (rgba[d] << 24) | (rgba[d + 1] << 16) | (rgba[d + 2] << 8) | rgba[d + 3];
    indices[i] = index.get(word) as number;
  }
  return { colourType: 3, bpp: 1, samples: indices, plte, trns };
}

// The image's samples in colour type `ct` (0, 2, 4 or 6) from its RGBA.
function samplesOf(rgba: Uint8Array, ct: number): Uint8Array {
  const n = rgba.length >>> 2;
  if (ct === 6) return rgba;
  const ch = ct === 0 ? 1 : ct === 4 ? 2 : 3;
  const out = new Uint8Array(n * ch);
  for (let i = 0, d = 0, s = 0; i < n; i++, d += 4, s += ch) {
    if (ct === 0) out[s] = rgba[d];
    else if (ct === 4) {
      out[s] = rgba[d];
      out[s + 1] = rgba[d + 3];
    } else {
      out[s] = rgba[d];
      out[s + 1] = rgba[d + 1];
      out[s + 2] = rgba[d + 2];
    }
  }
  return out;
}

// The channel form: grey / grey+alpha / RGB / RGBA, with a tRNS key when one
// transparent colour can stand for every transparent pixel.
function channelForm(rgba: Uint8Array, a: Analysis): Form {
  const key = a.opaque ? -1 : transparentKey(rgba, a.grey);
  let ct: number;
  let trns: Uint8Array | null = null;
  if (a.grey) {
    ct = a.opaque || key !== -1 ? 0 : 4;
    if (key !== -1) trns = Uint8Array.of(0, key);
  } else {
    ct = a.opaque || key !== -1 ? 2 : 6;
    if (key !== -1) trns = Uint8Array.of(0, key >>> 16, 0, (key >>> 8) & 0xff, 0, key & 0xff);
  }
  const bpp = ct === 0 ? 1 : ct === 4 ? 2 : ct === 2 ? 3 : 4;
  return { colourType: ct, bpp, samples: samplesOf(rgba, ct), plte: null, trns };
}

// All the forms that can hold this image exactly.
function formsOf(rgba: Uint8Array, a: Analysis): Form[] {
  const forms: Form[] = [];
  if (a.colours !== null) {
    forms.push(paletteForm(rgba, a.colours, 'count'));
    forms.push(paletteForm(rgba, a.colours, 'luminance'));
  }
  forms.push(channelForm(rgba, a));
  return forms;
}

// ---------------------------------------------------------------------------
// Per-row filter choice.
// ---------------------------------------------------------------------------

// How many filtered bytes a brute-force trial looks back over as dictionary.
const BRUTE_CONTEXT = 32768;

// Every row filtered by one fixed type.
function filterFixed(samples: Uint8Array, width: number, height: number, bpp: number, type: number): Uint8Array {
  const stride = width * bpp;
  const raw = new Uint8Array(height * (stride + 1));
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    raw[y * (stride + 1)] = type;
    filterRow(type, cur, prev, bpp, raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)));
  }
  return raw;
}

// Every row the filter whose signed byte sum is least (PNG spec's heuristic).
function filterMinSum(samples: Uint8Array, width: number, height: number, bpp: number): Uint8Array {
  const stride = width * bpp;
  const raw = new Uint8Array(height * (stride + 1));
  const trial: Uint8Array[] = [];
  for (let t = 0; t < 5; t++) trial.push(new Uint8Array(stride));
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    let type = 0;
    let best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t]);
      const s = signedSum(trial[t]);
      if (s < best) {
        best = s;
        type = t;
      }
    }
    const at = y * (stride + 1);
    raw[at] = type;
    raw.set(trial[type], at + 1);
  }
  return raw;
}

// Every row the filter holding the fewest distinct pairs of neighbouring
// bytes -- a cheap stand-in for how well the row will compress.
function filterPairs(samples: Uint8Array, width: number, height: number, bpp: number): Uint8Array {
  const stride = width * bpp;
  const raw = new Uint8Array(height * (stride + 1));
  const trial: Uint8Array[] = [];
  for (let t = 0; t < 5; t++) trial.push(new Uint8Array(stride));
  const seen = new Uint32Array(65536);
  let stamp = 0;
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    let type = 0;
    let best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t]);
      stamp++;
      let n = 0;
      const row = trial[t];
      for (let i = 1; i < row.length; i++) {
        const pair = (row[i - 1] << 8) | row[i];
        if (seen[pair] !== stamp) {
          seen[pair] = stamp;
          n++;
        }
      }
      if (n < best) {
        best = n;
        type = t;
      }
    }
    const at = y * (stride + 1);
    raw[at] = type;
    raw.set(trial[type], at + 1);
  }
  return raw;
}

// Every row the filter that compresses smallest, weighed by a short zlib run
// at level 4 over the row (with its filter byte) against the 32 KiB already
// written before it as dictionary -- the choice the row itself would make
// inside the real stream. The lower type wins a tie.
function filterBrute(samples: Uint8Array, width: number, height: number, bpp: number): Uint8Array {
  const stride = width * bpp;
  const raw = new Uint8Array(height * (stride + 1));
  const trial: Uint8Array[] = [];
  for (let t = 0; t < 5; t++) {
    const row = new Uint8Array(stride + 1);
    row[0] = t;
    trial.push(row);
  }
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * (stride + 1);
    const before = raw.subarray(Math.max(0, at - BRUTE_CONTEXT), at);
    let type = 0;
    let best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t].subarray(1));
      const cost = deflateSync(trial[t], before.length === 0 ? { level: 4, memLevel: 9 } : { level: 4, memLevel: 9, dictionary: before }).length;
      if (cost < best) {
        best = cost;
        type = t;
      }
    }
    raw.set(trial[type], at);
  }
  return raw;
}

// ---------------------------------------------------------------------------
// Compression: screen candidates cheaply, then write the winner hard.
// ---------------------------------------------------------------------------

// The bytes a form's PLTE and tRNS chunks add to its file.
function chunkBytes(form: Form): number {
  return (form.plte === null ? 0 : 12 + form.plte.length) + (form.trns === null ? 0 : 12 + form.trns.length);
}

// A quick screen of filtered data: zlib level 6, the level the contest's
// scale uses.
function screenBytes(data: Uint8Array): number {
  return deflateSync(data, { level: 6, memLevel: 8 }).length;
}

// The final stream for filtered data: level 9 under four settings -- the
// default and filtered strategies, each at memLevel 9 and 8 -- keeping
// whichever writes fewer bytes. memLevel changes zlib's match search, so a
// smaller table sometimes finds the shorter stream; the filtered strategy
// sometimes wins on filtered rows. All four are deterministic functions of
// the data, so the choice cannot differ between runs.
function finalStream(data: Uint8Array): Uint8Array {
  let best: Uint8Array | null = null;
  for (const memLevel of [9, 8]) {
    for (const strategy of [constants.Z_DEFAULT_STRATEGY, constants.Z_FILTERED]) {
      const stream = deflateSync(data, { level: 9, memLevel, strategy });
      if (best === null || stream.length < best.length) best = stream;
    }
  }
  return best as Uint8Array;
}

// The PNG of an image in `form` whose IDAT holds `idat`.
function pngOfForm(width: number, height: number, form: Form, idat: Uint8Array): Uint8Array {
  const chunks: Array<[string, Uint8Array]> = [['IHDR', ihdrOf(width, height, form.colourType)]];
  if (form.plte !== null) chunks.push(['PLTE', form.plte]);
  if (form.trns !== null) chunks.push(['tRNS', form.trns]);
  chunks.push(['IDAT', idat]);
  chunks.push(['IEND', new Uint8Array(0)]);
  return pngOf(chunks);
}

// Encode one decoded image: try every form with its filter choices, screen
// each at level 6, write the smallest with the final level-9 stream.
function encodeImage(image: DecodedImage): Uint8Array {
  const { width, height, rgba } = image;
  const a = analyse(rgba);
  const forms = formsOf(rgba, a);
  let best: { bytes: number; form: Form; data: Uint8Array } | null = null;
  for (const form of forms) {
    // Filter choices per form: every fixed filter, minimum-sum, fewest-pairs
    // and brute force. Fixed filters matter: on smooth photo-like pictures
    // no filter at all can beat every adaptive choice, so they are screened
    // like the rest instead of assumed worse.
    const choices: Array<'fixed0' | 'fixed1' | 'fixed2' | 'fixed3' | 'fixed4' | 'brute' | 'minsum' | 'pairs'> =
      ['fixed0', 'fixed1', 'fixed2', 'fixed3', 'fixed4', 'brute', 'minsum', 'pairs'];
    for (const choice of choices) {
      const data =
        choice === 'brute'
          ? filterBrute(form.samples, width, height, form.bpp)
          : choice === 'minsum'
            ? filterMinSum(form.samples, width, height, form.bpp)
            : choice === 'pairs'
              ? filterPairs(form.samples, width, height, form.bpp)
              : filterFixed(form.samples, width, height, form.bpp, Number(choice.slice(5)));
      const bytes = screenBytes(data) + chunkBytes(form);
      if (best === null || bytes < best.bytes) best = { bytes, form, data };
    }
  }
  if (best === null) throw new Error('no form for image');
  const idat = finalStream(best.data);
  return pngOfForm(width, height, best.form, idat);
}

// ---------------------------------------------------------------------------
// Entry point: answer every instance under /input/<id>/image.png at
// /output/<id>/image.png, creating each output directory itself.
// ---------------------------------------------------------------------------

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(INPUT_DIR).sort()) {
  const image = decodePng(new Uint8Array(readFileSync(join(INPUT_DIR, id, 'image.png'))));
  if (image.width !== 1280 || image.height !== 720) throw new Error(`${id}: not 1280x720`);
  const png = encodeImage(image);
  if (png.length > 4194304) throw new Error(`${id}: answer over 4 MiB`);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'image.png'), png);
}
