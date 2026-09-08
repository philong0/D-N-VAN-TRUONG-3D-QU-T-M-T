export const GNM_HEAD_V3_SCHEMA = {
  vertexCount: 17821,
  triangleCount: 35142,
} as const;

export interface GnmHeadRawData {
  positions: Float32Array;
  triangleIndices: Int32Array;
  triangleUvs: Float32Array;
  vertexCount: number;
  triangleCount: number;
}

export interface GnmIdentityBasisRawData {
  data: Float32Array;
  identityDim: number;
  vertexCount: number;
}

export interface GnmFaceMaskRawData {
  membership: Float32Array;
  vertexCount: number;
}

