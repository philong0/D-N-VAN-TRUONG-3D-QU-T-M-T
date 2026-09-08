export interface NpyArray {
  dtype: string;
  fortranOrder: boolean;
  shape: number[];
  data: Float32Array | Int32Array | Uint8Array;
}

export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
    view.getUint8(4),
    view.getUint8(5)
  );
  if (magic !== "\x93NUMPY") {
    throw new Error("Invalid NPY file format");
  }

  const major = view.getUint8(6);
  let headerLen = 0;
  let offset = 0;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    offset = 10;
  } else {
    headerLen = view.getUint32(8, true);
    offset = 12;
  }

  const headerStr = new TextDecoder().decode(new Uint8Array(buffer, offset, headerLen));
  const dataOffset = offset + headerLen;

  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const dtype = descrMatch ? descrMatch[1] : "<f4";

  const fortranMatch = headerStr.match(/'fortran_order':\s*(True|False)/);
  const fortranOrder = fortranMatch ? fortranMatch[1] === "True" : false;

  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
  let shape: number[] = [];
  if (shapeMatch && shapeMatch[1].trim()) {
    shape = shapeMatch[1]
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => parseInt(s, 10));
  }

  let data: Float32Array | Int32Array | Uint8Array;
  const dataSlice = buffer.slice(dataOffset);

  if (dtype === "<f4") {
    data = new Float32Array(dataSlice);
  } else if (dtype === "<i4") {
    data = new Int32Array(dataSlice);
  } else {
    data = new Uint8Array(dataSlice);
  }

  return { dtype, fortranOrder, shape, data };
}

