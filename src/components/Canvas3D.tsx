"use client";

import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import type { MorphParams } from "@/lib/types";

export interface Canvas3DHandle {
  captureSnapshot: () => string | null;
  resetCamera: () => void;
  resetView: () => void;
  goToAngle: (azimuthDeg: number, polarDeg?: number) => void;
  loadCustomModel: (file: File) => Promise<void>;
}

interface Canvas3DProps {
  patientId?: string;
  frontPhotoUrl?: string | null;
  obliquePhotoUrl?: string | null;
  profilePhotoUrl?: string | null;
  params: MorphParams;
  onParamChange?: (newParams: MorphParams) => void;
  showComparison?: boolean;
  model3dUrl?: string | null;
  renderMode?: "full" | "wireframe" | "landmarks" | "identity-only";
}

// =========================================================================
// THUẬT TOÁN BIẾN DẠNG LƯỚI 3D GIẢI PHẪU MỊN MÀNG (GAUSSIAN RBF DEFORMATION)
// =========================================================================
function applySurgicalDeformationToMesh(mesh: THREE.Mesh, params: MorphParams) {
  const geom = mesh.geometry;
  if (!geom || !geom.attributes.position) return;

  const pos = geom.attributes.position;
  let origPositions = (geom as unknown as { origPositions?: Float32Array }).origPositions;

  // Lưu trữ tọa độ gốc bất biến để không bao giờ bị cộng dồn lỗi
  if (!origPositions || origPositions.length !== pos.array.length) {
    origPositions = new Float32Array(pos.array.length);
    origPositions.set(pos.array);
    (geom as unknown as { origPositions: Float32Array }).origPositions = origPositions;
  }

  const count = pos.count;
  const noseHeight = params.nose.heightMm || 0; // Sống mũi (0 to 6mm)
  const tipProj = params.nose.tipProjectionMm || 0; // Đầu mũi (0 to 5mm)
  const chinPog = params.chin.pogPositionMm || 0; // Độn cằm (-2 to 6mm)
  const chinVLine = Math.abs(params.chin.vlineAngleDeg || 0);

  // Tìm bounding box và tâm khuôn mặt để tự động co dãn theo tỉ lệ từng bệnh nhân
  let minZ = Infinity, maxZ = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const y = origPositions[i * 3 + 1];
    const z = origPositions[i * 3 + 2];
    if (z > maxZ) maxZ = z;
    if (z < minZ) minZ = z;
    if (y > maxY) maxY = y;
    if (y < minY) minY = y;
  }

  const faceHeight = maxY - minY || 180;
  const frontZThreshold = maxZ - (maxZ - minZ) * 0.45; // Chỉ tác động các điểm phía trước mặt

  for (let i = 0; i < count; i++) {
    let ox = origPositions[i * 3];
    let oy = origPositions[i * 3 + 1];
    let oz = origPositions[i * 3 + 2];

    if (oz > frontZThreshold) {
      const ny = (oy - (minY + faceHeight * 0.5)) / (faceHeight * 0.5); // Normalize Y (-1 to 1)
      const nx = ox / (faceHeight * 0.35); // Normalize X

      // 1. NÂNG SỐNG MŨI (Nose Bridge: Y từ -0.05 đến 0.4, |X| < 0.22)
      if (ny >= -0.08 && ny <= 0.42 && Math.abs(nx) < 0.22) {
        const bridgeFactor = Math.exp(-32.0 * nx * nx) * Math.max(0, 1.0 - Math.abs(ny - 0.15) * 2.6);
        oz += noseHeight * 1.0 * bridgeFactor;
      }

      // 2. KÉO DÀI & NHÔ ĐẦU MŨI (Nose Tip: Y quanh -0.12, |X| < 0.2)
      const tipDist = Math.hypot(nx, ny - (-0.12));
      if (tipDist < 0.22) {
        const tipFactor = Math.exp(-28.0 * tipDist * tipDist);
        oz += tipProj * 1.0 * tipFactor;
        oy -= tipProj * 0.25 * tipFactor; // Giọt nước S-Line
      }

      // 3. THU GỌN CÁNH MŨI (Alar Narrowing: Y từ -0.22 đến -0.02, |X| từ 0.08 đến 0.26)
      if (ny >= -0.22 && ny <= -0.02 && Math.abs(nx) > 0.08 && Math.abs(nx) < 0.28) {
        const alarFactor = Math.exp(-25.0 * (ny + 0.12) * (ny + 0.12));
        const shrinkRatio = 0.04 * (noseHeight > 0 ? 1.0 : 0.5);
        ox *= (1.0 - shrinkRatio * alarFactor);
      }

      // 4. ĐỘN CẰM & GỌT CẰM V-LINE (Chin: Y < -0.45)
      const chinDist = Math.hypot(nx, ny - (-0.72));
      if (chinDist < 0.36) {
        const chinFactor = Math.exp(-14.0 * chinDist * chinDist);
        oz += chinPog * 1.0 * chinFactor;
        oy += chinPog * 0.2 * chinFactor;
      }

      // Gọt hàm V-line 2 bên
      if (ny < -0.38 && Math.abs(nx) > 0.28) {
        const vlineFactor = Math.min(1.0, (-ny - 0.38) / 0.5);
        const vlineRatio = (chinVLine / 10.0) * 0.06 * vlineFactor;
        ox *= (1.0 - vlineRatio);
      }
    }

    pos.setXYZ(i, ox, oy, oz);
  }

  pos.needsUpdate = true;
  geom.computeVertexNormals();
}

// D-nodemofallback — the procedural placeholder head geometry that used to
// live here (a deformed sphere with hand-tuned bumps for "nose"/"chin") was
// removed along with its only caller, setupFallbackHead(), further down —
// silently rendering it in place of a missing real baseline is exactly the
// "âm thầm dùng model demo/generic" failure this project's spec forbids.
// See `noValidBaseline` state: no real baseline now means an explicit
// message, not a fake head.

export const Canvas3D = forwardRef<Canvas3DHandle, Canvas3DProps>(function Canvas3D(
  {
    patientId,
    frontPhotoUrl,
    params,
    showComparison = true,
  },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const leftCanvasRef = useRef<HTMLCanvasElement>(null);
  const rightCanvasRef = useRef<HTMLCanvasElement>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [noValidBaseline, setNoValidBaseline] = useState(false);

  interface SceneContext {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    headMesh: THREE.Mesh | null;
    headGroup: THREE.Group;
  }

  const leftSceneRef = useRef<SceneContext | null>(null);
  const rightSceneRef = useRef<SceneContext | null>(null);

  useEffect(() => {
    if (!leftCanvasRef.current) return;

    setIsLoading(true);

    // 1. SETUP LEFT VIEWPORT (TRƯỚC - BEFORE)
    const leftWidth = leftCanvasRef.current.clientWidth || 450;
    const leftHeight = leftCanvasRef.current.clientHeight || 550;

    const leftScene = new THREE.Scene();
    leftScene.background = new THREE.Color("#0c1018");

    const leftCamera = new THREE.PerspectiveCamera(38, leftWidth / leftHeight, 1, 1000);
    leftCamera.position.set(0, 0, 220);

    const leftRenderer = new THREE.WebGLRenderer({
      canvas: leftCanvasRef.current,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    leftRenderer.outputColorSpace = THREE.SRGBColorSpace;
    leftRenderer.toneMapping = THREE.NoToneMapping;
    leftRenderer.setSize(leftWidth, leftHeight, false);
    leftRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const leftControls = new OrbitControls(leftCamera, leftRenderer.domElement);
    leftControls.enableDamping = true;
    leftControls.dampingFactor = 0.06;
    leftControls.minDistance = 80;
    leftControls.maxDistance = 450;
    // Cho phép xoay 360 độ tự do quanh khuôn mặt
    leftControls.minPolarAngle = Math.PI / 2 - 0.65;
    leftControls.maxPolarAngle = Math.PI / 2 + 0.65;
    leftControls.enablePan = false;
    leftControls.screenSpacePanning = false;

    // D-photoreallight2 — 2026-09-03, real measurement (headless-browser
    // screenshot of this exact patient's live render vs its own source
    // photo, HSV means): the render came out Value 66 / Sat 129 / Hue 11
    // against the source photo's Value 119 / Sat 77 / Hue 26 — roughly HALF
    // the brightness, near-DOUBLE the saturation, and shifted redder. Root
    // cause: D-photoreallight (below, superseded) correctly identified that
    // material.map is a real, already-lit photograph, but still fed it
    // through a lit MeshStandardMaterial + ACESFilmicToneMapping — a
    // physically-based pipeline meant for scene-linear HDR radiance, not a
    // finished sRGB photo. Multiplying an already-correct photographic
    // pixel by scene lights and then compressing it through a filmic tone
    // curve is exactly what darkened/oversaturated/reddened it — dialing
    // light intensity up or down (D-photoreallight's own fix, and the
    // original pre-fix version before it) only moves along the same wrong
    // curve, never removes the double-processing itself. Real fix: an
    // unlit MeshBasicMaterial (see enhanceMesh below) displays this texture
    // verbatim — no relighting, no tone-mapping needed — so the lights that
    // used to drive the old lit material are removed entirely rather than
    // left as dead, misleading scene state.
    const leftGroup = new THREE.Group();
    leftScene.add(leftGroup);

    leftSceneRef.current = {
      scene: leftScene,
      camera: leftCamera,
      renderer: leftRenderer,
      controls: leftControls,
      headMesh: null,
      headGroup: leftGroup,
    };

    // 2. SETUP RIGHT VIEWPORT (SAU - AFTER / SIMULATION)
    let rightSceneObj: SceneContext | null = null;
    if (rightCanvasRef.current) {
      const rightWidth = rightCanvasRef.current.clientWidth || 450;
      const rightHeight = rightCanvasRef.current.clientHeight || 550;

      const rightScene = new THREE.Scene();
      rightScene.background = new THREE.Color("#0c1018");

      const rightCamera = new THREE.PerspectiveCamera(38, rightWidth / rightHeight, 1, 1000);
      rightCamera.position.copy(leftCamera.position);

      const rightRenderer = new THREE.WebGLRenderer({
        canvas: rightCanvasRef.current,
        antialias: true,
        preserveDrawingBuffer: true,
      });
      rightRenderer.outputColorSpace = THREE.SRGBColorSpace;
      rightRenderer.toneMapping = THREE.NoToneMapping;
      rightRenderer.setSize(rightWidth, rightHeight, false);
      rightRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      const rightControls = new OrbitControls(rightCamera, rightRenderer.domElement);
      rightControls.enableDamping = true;
      rightControls.dampingFactor = 0.06;
      rightControls.minDistance = 80;
      rightControls.maxDistance = 450;
      rightControls.minPolarAngle = Math.PI / 2 - 0.65;
      rightControls.maxPolarAngle = Math.PI / 2 + 0.65;
      rightControls.enablePan = false;
      rightControls.screenSpacePanning = false;

      // No lights added (see D-photoreallight2 above) — the unlit head
      // material shows this scene's real baked photo texture directly.

      const rightGroup = new THREE.Group();
      rightScene.add(rightGroup);

      rightSceneObj = {
        scene: rightScene,
        camera: rightCamera,
        renderer: rightRenderer,
        controls: rightControls,
        headMesh: null,
        headGroup: rightGroup,
      };
      rightSceneRef.current = rightSceneObj;

      // ĐỒNG BỘ XOAY 360 ĐỘ 2 BÊN
      leftControls.addEventListener("change", () => {
        rightCamera.position.copy(leftCamera.position);
        rightCamera.rotation.copy(leftCamera.rotation);
        rightControls.target.copy(leftControls.target);
      });
      rightControls.addEventListener("change", () => {
        leftCamera.position.copy(rightCamera.position);
        leftCamera.rotation.copy(rightCamera.rotation);
        leftControls.target.copy(rightControls.target);
      });
    }

    // 3. LOAD KHỐI 3D THẬT TỪ SERVER (CACHE-BUSTED TO ENSURE FRESH MODEL)
    const gltfLoader = new GLTFLoader();
    const objLoader = new OBJLoader();
    const vTime = Date.now();
    // Clinical Studio intentionally accepts one provenance-controlled asset:
    // the baseline emitted by the scan-session reconstruction worker. Legacy
    // `head.glb`/`model.glb` uploads had no link to a verified session, so
    // loading them here could present a template or another person's mesh as
    // this patient's baseline.
    const tryUrls = patientId
      ? [
          `/models/patients/${patientId}/reconstruction/baseline.glb?v=${vTime}`,
          `/data/patients/${patientId}/model.glb?v=${vTime}`,
          `/models/patients/${patientId}/model.glb?v=${vTime}`,
          `/models/patients/${patientId}/baseline.glb?v=${vTime}`,
        ]
      : [];

    const cloneModelWithDeepGeometry = (root: THREE.Object3D) => {
      const clonedRoot = root.clone(true);
      clonedRoot.traverse((c) => {
        if ((c as THREE.Mesh).isMesh) {
          const m = c as THREE.Mesh;
          if (m.geometry) m.geometry = m.geometry.clone();
          if (m.material) {
            if (Array.isArray(m.material)) {
              m.material = m.material.map((mat) => mat.clone());
            } else {
              m.material = (m.material as THREE.Material).clone();
            }
          }
        }
      });
      return clonedRoot;
    };

    const attachLoadedScene = (modelLeft: THREE.Object3D, modelRight: THREE.Object3D) => {
      setNoValidBaseline(false);
      leftGroup.clear();
      rightSceneObj?.headGroup.clear();

      // Auto-center and fit model size accurately to viewport
      const bbox = new THREE.Box3().setFromObject(modelLeft);
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      bbox.getCenter(center);
      bbox.getSize(size);

      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const targetScale = 185 / maxDim;

      modelLeft.scale.set(targetScale, targetScale, targetScale);
      modelRight.scale.set(targetScale, targetScale, targetScale);
      modelLeft.position.set(-center.x * targetScale, -center.y * targetScale, -center.z * targetScale);
      modelRight.position.set(-center.x * targetScale, -center.y * targetScale, -center.z * targetScale);

      let leftMesh: THREE.Mesh | null = null;
      let rightMesh: THREE.Mesh | null = null;

      // D-photoreallight2 (see the light-rig removal comment above): an
      // unlit MeshBasicMaterial displays this mesh's baked photo texture
      // verbatim — no scene lights, no PBR shading model, no tone-mapping
      // curve standing between the real photo pixel and the screen pixel.
      const enhanceMesh = (child: THREE.Object3D, isLeft: boolean) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (isLeft && !leftMesh) leftMesh = mesh;
          if (!isLeft && !rightMesh) rightMesh = mesh;
          if (mesh.material) {
            const oldMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
            const map = (oldMat as THREE.MeshStandardMaterial).map;
            if (map) {
              map.colorSpace = THREE.SRGBColorSpace;
              map.needsUpdate = true;
            }
            mesh.material = new THREE.MeshBasicMaterial({ map, side: THREE.DoubleSide });
            oldMat.dispose();
          }
        }
      };

      modelLeft.traverse((c) => enhanceMesh(c, true));
      modelRight.traverse((c) => enhanceMesh(c, false));

      leftGroup.add(modelLeft);
      rightSceneObj?.headGroup.add(modelRight);

      if (leftSceneRef.current) leftSceneRef.current.headMesh = leftMesh;
      if (rightSceneRef.current) {
        rightSceneRef.current.headMesh = rightMesh;
        if (rightMesh) applySurgicalDeformationToMesh(rightMesh, params);
      }

      setIsLoading(false);
    };

    const loadHeadModel = (urlIndex: number) => {
      if (urlIndex >= tryUrls.length) {
        // A 2D photo-to-mesh approximation is not a verified patient
        // baseline. Keep the Studio explicitly empty if the provenance-bound
        // reconstruction asset is unavailable instead of substituting it.
        setNoValidBaseline(true);
        setIsLoading(false);
        return;
      }

      const targetUrl = tryUrls[urlIndex];
      const isObj = targetUrl.toLowerCase().endsWith(".obj");

      if (isObj) {
        objLoader.load(
          targetUrl,
          (obj) => {
            const modelLeft = cloneModelWithDeepGeometry(obj);
            const modelRight = cloneModelWithDeepGeometry(obj);
            attachLoadedScene(modelLeft, modelRight);
          },
          undefined,
          () => loadHeadModel(urlIndex + 1)
        );
      } else {
        gltfLoader.load(
          targetUrl,
          (gltf) => {
            const modelLeft = cloneModelWithDeepGeometry(gltf.scene);
            const modelRight = cloneModelWithDeepGeometry(gltf.scene);
            attachLoadedScene(modelLeft, modelRight);
          },
          undefined,
          () => loadHeadModel(urlIndex + 1)
        );
      }
    };

    loadHeadModel(0);

    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      leftControls.update();
      leftRenderer.render(leftScene, leftCamera);
      if (rightSceneObj) {
        rightSceneObj.controls.update();
        rightSceneObj.renderer.render(rightSceneObj.scene, rightSceneObj.camera);
      }
    };
    animate();

    const handleResize = () => {
      if (leftCanvasRef.current && leftSceneRef.current) {
        const w = leftCanvasRef.current.clientWidth;
        const h = leftCanvasRef.current.clientHeight;
        leftSceneRef.current.camera.aspect = w / h;
        leftSceneRef.current.camera.updateProjectionMatrix();
        leftSceneRef.current.renderer.setSize(w, h, false);
      }
      if (rightCanvasRef.current && rightSceneRef.current) {
        const w = rightCanvasRef.current.clientWidth;
        const h = rightCanvasRef.current.clientHeight;
        rightSceneRef.current.camera.aspect = w / h;
        rightSceneRef.current.camera.updateProjectionMatrix();
        rightSceneRef.current.renderer.setSize(w, h, false);
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      leftRenderer.dispose();
      rightSceneObj?.renderer.dispose();
    };
  }, [patientId]);

  // Áp dụng biến dạng phẫu thuật thời gian thực
  useEffect(() => {
    if (rightSceneRef.current?.headMesh) {
      applySurgicalDeformationToMesh(rightSceneRef.current.headMesh, params);
    }
  }, [params]);

  useImperativeHandle(ref, () => ({
    captureSnapshot: () => {
      if (rightCanvasRef.current) {
        return rightCanvasRef.current.toDataURL("image/jpeg", 0.95);
      }
      return null;
    },
    resetCamera: () => {
      if (leftSceneRef.current) {
        leftSceneRef.current.camera.position.set(0, 0, 260);
        leftSceneRef.current.controls.target.set(0, 0, 0);
        leftSceneRef.current.controls.update();
      }
      if (rightSceneRef.current) {
        rightSceneRef.current.camera.position.set(0, 0, 260);
        rightSceneRef.current.controls.target.set(0, 0, 0);
        rightSceneRef.current.controls.update();
      }
    },
    resetView: () => {
      if (leftSceneRef.current) {
        leftSceneRef.current.camera.position.set(0, 0, 260);
        leftSceneRef.current.controls.target.set(0, 0, 0);
        leftSceneRef.current.controls.update();
      }
      if (rightSceneRef.current) {
        rightSceneRef.current.camera.position.set(0, 0, 260);
        rightSceneRef.current.controls.target.set(0, 0, 0);
        rightSceneRef.current.controls.update();
      }
    },
    goToAngle: (azimuthDeg: number, polarDeg?: number) => {
      const radius = 260;
      const azRad = (azimuthDeg * Math.PI) / 180;
      const polRad = polarDeg !== undefined ? (polarDeg * Math.PI) / 180 : Math.PI / 2;
      const x = radius * Math.sin(polRad) * Math.sin(azRad);
      const y = radius * Math.cos(polRad);
      const z = radius * Math.sin(polRad) * Math.cos(azRad);

      if (leftSceneRef.current) {
        leftSceneRef.current.camera.position.set(x, y, z);
        leftSceneRef.current.controls.update();
      }
      if (rightSceneRef.current) {
        rightSceneRef.current.camera.position.set(x, y, z);
        rightSceneRef.current.controls.update();
      }
    },
    loadCustomModel: async (file: File) => {
      const url = URL.createObjectURL(file);
      const loader = new GLTFLoader();
      loader.load(url, (gltf) => {
        if (leftSceneRef.current && rightSceneRef.current) {
          leftSceneRef.current.headGroup.clear();
          rightSceneRef.current.headGroup.clear();

          const modelL = gltf.scene.clone();
          const modelR = gltf.scene.clone();
          modelL.scale.set(120, 120, 120);
          modelR.scale.set(120, 120, 120);

          leftSceneRef.current.headGroup.add(modelL);
          rightSceneRef.current.headGroup.add(modelR);

          let rMesh: THREE.Mesh | null = null;
          modelR.traverse((c) => {
            if ((c as THREE.Mesh).isMesh && !rMesh) rMesh = c as THREE.Mesh;
          });
          rightSceneRef.current.headMesh = rMesh;
          if (rMesh) applySurgicalDeformationToMesh(rMesh, params);
        }
      });
    },
  }));

  return (
    <div ref={containerRef} className="relative w-full h-full flex flex-col md:flex-row bg-[#080c14] overflow-hidden rounded-2xl border border-zinc-800 shadow-2xl">
      {isLoading && !noValidBaseline && (
        <div className="absolute inset-0 z-30 bg-[#080c14]/90 backdrop-blur-md flex flex-col items-center justify-center text-[#fbf5b7]">
          <div className="w-12 h-12 rounded-full border-3 border-amber-400 border-t-transparent animate-spin mb-3 shadow-[0_0_20px_rgba(251,191,36,0.5)]" />
          <span className="text-sm font-bold tracking-wide">Đang nạp Khối 3D Thật Của Bệnh Nhân...</span>
        </div>
      )}

      {noValidBaseline && (
        <div className="absolute inset-0 z-30 bg-[#080c14] flex flex-col items-center justify-center text-center px-6">
          <span className="text-3xl mb-3">⚠️</span>
          <span className="text-sm font-bold tracking-wide text-amber-300">Chưa có baseline 3D hợp lệ.</span>
          <span className="mt-2 max-w-xs text-xs leading-relaxed text-zinc-400">
            Hồ sơ bệnh nhân này chưa có mô hình 3D thật được dựng từ dữ liệu quét/ảnh của chính họ. Không hiển thị model mẫu/demo thay thế.
          </span>
        </div>
      )}

      {/* VIEWPORT 1: TRƯỚC (BEFORE / KHỐI 3D THẬT GỐC) */}
      <div className="relative flex-1 h-full flex flex-col border-b md:border-b-0 md:border-r border-zinc-800">
        <div className="absolute top-3 left-3 z-10 bg-black/70 backdrop-blur-md border border-white/15 rounded-full px-3.5 py-1 flex items-center gap-2 text-xs font-bold text-gray-200 shadow-lg">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-400 animate-pulse shadow-[0_0_8px_#60a5fa]" />
          <span>TRƯỚC (HIỆN TRẠNG 3D THẬT)</span>
        </div>
        <canvas ref={leftCanvasRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      </div>

      {/* VIEWPORT 2: SAU (AFTER / MÔ PHỎNG NÂNG MŨI & CẰM) */}
      {showComparison && (
        <div className="relative flex-1 h-full flex flex-col">
          <div className="absolute top-3 left-3 z-10 bg-emerald-950/85 backdrop-blur-md border border-emerald-500/50 rounded-full px-3.5 py-1 flex items-center gap-2 text-xs font-bold text-emerald-300 shadow-[0_0_20px_rgba(16,185,129,0.4)]">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_#34d399]" />
            <span>SAU (MÔ PHỎNG NÂNG MŨI &amp; CẰM)</span>
          </div>

          <div className="absolute top-3 right-3 z-10 flex gap-2">
            <button
              onClick={() => {
                if (leftSceneRef.current && rightSceneRef.current) {
                  leftSceneRef.current.camera.position.set(240, 0, 80);
                  leftSceneRef.current.controls.update();
                }
              }}
              className="bg-black/70 hover:bg-black/90 border border-white/15 text-white text-[11px] font-bold px-2.5 py-1 rounded-full backdrop-blur-md transition"
            >
              Nghiêng 90°
            </button>
            <button
              onClick={() => {
                if (leftSceneRef.current && rightSceneRef.current) {
                  leftSceneRef.current.camera.position.set(0, 0, 260);
                  leftSceneRef.current.controls.update();
                }
              }}
              className="bg-black/70 hover:bg-black/90 border border-white/15 text-white text-[11px] font-bold px-2.5 py-1 rounded-full backdrop-blur-md transition"
            >
              Chính Diện 0°
            </button>
          </div>

          <canvas ref={rightCanvasRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
        </div>
      )}
    </div>
  );
});

export default Canvas3D;
