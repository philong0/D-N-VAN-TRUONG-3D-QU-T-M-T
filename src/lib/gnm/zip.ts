/**
 * Lightweight ZIP entry extractor for .npz (uncompressed ZIP) archives.
 */
export async function extractZipEntries(
  buffer: ArrayBuffer,
  wantedEntries: readonly string[]
): Promise<Map<string, ArrayBuffer>> {
  const result = new Map<string, ArrayBuffer>();
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  let offset = 0;
  while (offset < bytes.length - 4) {
    const sig = view.getUint32(offset, true);
    if (sig !== 0x04034b50) {
      break; // End of local file headers
    }

    const compressedSize = view.getUint32(offset + 18, true);
    const uncompressedSize = view.getUint32(offset + 22, true);
    const fileNameLen = view.getUint16(offset + 26, true);
    const extraLen = view.getUint16(offset + 28, true);

    const fileNameBytes = bytes.subarray(offset + 30, offset + 30 + fileNameLen);
    const fileName = new TextDecoder().decode(fileNameBytes);

    const dataOffset = offset + 30 + fileNameLen + extraLen;

    if (wantedEntries.includes(fileName)) {
      const entryData = buffer.slice(dataOffset, dataOffset + (uncompressedSize || compressedSize));
      result.set(fileName, entryData);
    }

    offset = dataOffset + (compressedSize || uncompressedSize);
  }

  return result;
}

