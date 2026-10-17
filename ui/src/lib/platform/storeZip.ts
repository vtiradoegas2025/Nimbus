const LOCAL_FILE_HEADER = 0x04034b50;
const CENTRAL_FILE_HEADER = 0x02014b50;
const END_OF_CENTRAL_DIRECTORY = 0x06054b50;
const UTF8_FLAG = 0x0800;
const MAX_ARCHIVE_BYTES = 96 * 1024 * 1024;
const MAX_ARCHIVE_FILES = 32;

export interface StoreZipEntry { name: string; data: Uint8Array }

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function u16(view: DataView, offset: number, value: number): number {
  view.setUint16(offset, value, true); return offset + 2;
}

function u32(view: DataView, offset: number, value: number): number {
  view.setUint32(offset, value >>> 0, true); return offset + 4;
}

function safeName(name: string): boolean {
  return name.length > 0 && name.length <= 240 && !name.startsWith("/")
    && !name.includes("\\") && !name.split("/").includes("..");
}

/** Minimal deterministic ZIP writer. Entries are stored without compression so
 * publication packages remain dependency-free and auditable by standard tools. */
export function createStoreZip(entries: readonly StoreZipEntry[]): Uint8Array {
  if (entries.length === 0 || entries.length > MAX_ARCHIVE_FILES) throw new Error("ZIP entry count is outside the supported publication bound.");
  const encoder = new TextEncoder();
  const names = entries.map((entry) => encoder.encode(entry.name));
  const unique = new Set(entries.map((entry) => entry.name));
  if (unique.size !== entries.length || entries.some((entry) => !safeName(entry.name))) throw new Error("ZIP entry names must be unique safe relative paths.");
  const locals = entries.reduce((total, entry, index) => total + 30 + names[index].length + entry.data.length, 0);
  const central = entries.reduce((total, _entry, index) => total + 46 + names[index].length, 0);
  const total = locals + central + 22;
  if (!Number.isSafeInteger(total) || total > MAX_ARCHIVE_BYTES) throw new Error("Publication archive exceeds the 96 MiB bound.");
  const output = new Uint8Array(total);
  const view = new DataView(output.buffer);
  const records: Array<{ offset: number; crc: number }> = [];
  let offset = 0;
  entries.forEach((entry, index) => {
    const localOffset = offset;
    const checksum = crc32(entry.data);
    offset = u32(view, offset, LOCAL_FILE_HEADER);
    offset = u16(view, offset, 20); offset = u16(view, offset, UTF8_FLAG); offset = u16(view, offset, 0);
    offset = u16(view, offset, 0); offset = u16(view, offset, 0); offset = u32(view, offset, checksum);
    offset = u32(view, offset, entry.data.length); offset = u32(view, offset, entry.data.length);
    offset = u16(view, offset, names[index].length); offset = u16(view, offset, 0);
    output.set(names[index], offset); offset += names[index].length;
    output.set(entry.data, offset); offset += entry.data.length;
    records.push({ offset: localOffset, crc: checksum });
  });
  const centralOffset = offset;
  entries.forEach((entry, index) => {
    offset = u32(view, offset, CENTRAL_FILE_HEADER);
    offset = u16(view, offset, 20); offset = u16(view, offset, 20); offset = u16(view, offset, UTF8_FLAG); offset = u16(view, offset, 0);
    offset = u16(view, offset, 0); offset = u16(view, offset, 0); offset = u32(view, offset, records[index].crc);
    offset = u32(view, offset, entry.data.length); offset = u32(view, offset, entry.data.length);
    offset = u16(view, offset, names[index].length); offset = u16(view, offset, 0); offset = u16(view, offset, 0);
    offset = u16(view, offset, 0); offset = u16(view, offset, 0); offset = u32(view, offset, 0); offset = u32(view, offset, records[index].offset);
    output.set(names[index], offset); offset += names[index].length;
  });
  const centralSize = offset - centralOffset;
  offset = u32(view, offset, END_OF_CENTRAL_DIRECTORY);
  offset = u16(view, offset, 0); offset = u16(view, offset, 0);
  offset = u16(view, offset, entries.length); offset = u16(view, offset, entries.length);
  offset = u32(view, offset, centralSize); offset = u32(view, offset, centralOffset); u16(view, offset, 0);
  return output;
}

/** Read only the bounded, uncompressed ZIP subset emitted above. Unsupported
 * compression, descriptors, unsafe paths, corruption, and duplicates fail closed. */
export function readStoreZip(bytes: Uint8Array): ReadonlyMap<string, Uint8Array> {
  if (bytes.length < 22 || bytes.length > MAX_ARCHIVE_BYTES) throw new Error("Figure package size is outside the supported bound.");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const decoder = new TextDecoder();
  const files = new Map<string, Uint8Array>();
  const localRecords = new Map<string, { offset: number; crc: number; size: number }>();
  let offset = 0;
  while (offset + 4 <= bytes.length && view.getUint32(offset, true) === LOCAL_FILE_HEADER) {
    if (files.size >= MAX_ARCHIVE_FILES || offset + 30 > bytes.length) throw new Error("Figure package has too many or truncated entries.");
    const flags = view.getUint16(offset + 6, true);
    const method = view.getUint16(offset + 8, true);
    const expectedCrc = view.getUint32(offset + 14, true);
    const compressedSize = view.getUint32(offset + 18, true);
    const size = view.getUint32(offset + 22, true);
    const nameLength = view.getUint16(offset + 26, true);
    const extraLength = view.getUint16(offset + 28, true);
    if ((flags & ~UTF8_FLAG) !== 0 || method !== 0 || compressedSize !== size) throw new Error("Figure package uses an unsupported ZIP feature.");
    const nameStart = offset + 30;
    const dataStart = nameStart + nameLength + extraLength;
    const dataEnd = dataStart + size;
    if (nameLength === 0 || dataEnd > bytes.length) throw new Error("Figure package entry is truncated.");
    const name = decoder.decode(bytes.subarray(nameStart, nameStart + nameLength));
    if (!safeName(name) || files.has(name)) throw new Error("Figure package contains an unsafe or duplicate path.");
    const data = bytes.slice(dataStart, dataEnd);
    if (crc32(data) !== expectedCrc) throw new Error(`Figure package entry '${name}' failed CRC validation.`);
    files.set(name, data);
    localRecords.set(name, { offset, crc: expectedCrc, size });
    offset = dataEnd;
  }
  if (files.size === 0 || offset + 4 > bytes.length || view.getUint32(offset, true) !== CENTRAL_FILE_HEADER) {
    throw new Error("Figure package has no valid ZIP directory.");
  }
  const centralOffset = offset;
  for (let index = 0; index < files.size; index += 1) {
    if (offset + 46 > bytes.length || view.getUint32(offset, true) !== CENTRAL_FILE_HEADER) throw new Error("Figure package central directory is truncated.");
    const flags = view.getUint16(offset + 8, true); const method = view.getUint16(offset + 10, true);
    const checksum = view.getUint32(offset + 16, true); const compressedSize = view.getUint32(offset + 20, true);
    const size = view.getUint32(offset + 24, true); const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true); const commentLength = view.getUint16(offset + 32, true);
    const localOffset = view.getUint32(offset + 42, true); const nameStart = offset + 46;
    const end = nameStart + nameLength + extraLength + commentLength;
    if ((flags & ~UTF8_FLAG) !== 0 || method !== 0 || compressedSize !== size || nameLength === 0 || end > bytes.length) throw new Error("Figure package central entry is invalid.");
    const name = decoder.decode(bytes.subarray(nameStart, nameStart + nameLength));
    const local = localRecords.get(name);
    if (!local || local.offset !== localOffset || local.crc !== checksum || local.size !== size) throw new Error("Figure package central directory does not match its local entries.");
    offset = end;
  }
  const centralSize = offset - centralOffset;
  if (offset + 22 !== bytes.length || view.getUint32(offset, true) !== END_OF_CENTRAL_DIRECTORY
    || view.getUint16(offset + 4, true) !== 0 || view.getUint16(offset + 6, true) !== 0
    || view.getUint16(offset + 8, true) !== files.size || view.getUint16(offset + 10, true) !== files.size
    || view.getUint32(offset + 12, true) !== centralSize || view.getUint32(offset + 16, true) !== centralOffset
    || view.getUint16(offset + 20, true) !== 0) throw new Error("Figure package ZIP directory terminator is invalid.");
  return files;
}
