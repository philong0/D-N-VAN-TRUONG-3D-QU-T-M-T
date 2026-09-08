"use client";

import { forwardRef, useImperativeHandle, useRef } from "react";
import Canvas3D, { type Canvas3DHandle } from "@/components/Canvas3D";
import type { MorphGroup } from "@/lib/services-catalog";
import type { MorphParams } from "@/lib/types";

interface SplitCompareProps {
  photoUrl: string | null;
  obliquePhotoUrl?: string | null;
  profilePhotoUrl?: string | null;
  belowPhotoUrl?: string | null;
  patientId?: string;
  morph: MorphParams;
  unlockedGroups: Set<MorphGroup>;
  afterImageUrl?: string | null;
  autoRotate?: boolean;
  renderMode?: "full" | "identity-only";
}

const SplitCompare = forwardRef<Canvas3DHandle, SplitCompareProps>(function SplitCompare(
  {
    photoUrl,
    obliquePhotoUrl,
    profilePhotoUrl,
    patientId,
    morph,
  },
  ref
) {
  const canvasRef = useRef<Canvas3DHandle>(null);

  useImperativeHandle(ref, () => ({
    captureSnapshot: () => canvasRef.current?.captureSnapshot() ?? null,
    resetCamera: () => canvasRef.current?.resetCamera(),
    resetView: () => canvasRef.current?.resetView(),
    goToAngle: (azimuthDeg: number, polarDeg?: number) => canvasRef.current?.goToAngle(azimuthDeg, polarDeg),
    loadCustomModel: async (file: File) => canvasRef.current?.loadCustomModel(file),
  }));

  return (
    <div className="relative w-full h-full min-h-[500px]">
      <Canvas3D
        ref={canvasRef}
        patientId={patientId}
        frontPhotoUrl={photoUrl}
        obliquePhotoUrl={obliquePhotoUrl}
        profilePhotoUrl={profilePhotoUrl}
        params={morph}
        showComparison={true}
      />
    </div>
  );
});

export default SplitCompare;
