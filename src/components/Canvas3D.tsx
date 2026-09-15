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

export interface SkinToneConfig {
  brightness: number; // 0.8 to 1.35, default 1.05
  warmth: number;     // -20 to +20, default 0
  smoothness: number; // 0.5 to 1.0, default 0.82
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
  skinTone?: SkinToneConfig;
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
  const noseHeight = params?.nose?.heightMm || 0; // Sống mũi (0 to 6mm)
  const tipProj = params?.nose?.tipProjectionMm || 0; // Đầu mũi (0 to 5mm)
  const chinPog = params?.chin?.pogPositionMm || 0; // Độn cằm (-2 to 6mm)
  const chinVLine = Math.abs(params?.chin?.vlineAngleDeg || 0);

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
  const mmScale = faceHeight < 2.0 ? 0.001 : 1.0;
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
        oz += noseHeight * mmScale * bridgeFactor;
      }

      // 2. KÉO DÀI & NHÔ ĐẦU MŨI (Nose Tip: Y quanh -0.12, |X| < 0.2)
      const tipDist = Math.hypot(nx, ny - (-0.12));
      if (tipDist < 0.22) {
        const tipFactor = Math.exp(-28.0 * tipDist * tipDist);
        oz += tipProj * mmScale * tipFactor;
        oy -= tipProj * mmScale * 0.25 * tipFactor; // Giọt nước S-Line
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
        oz += chinPog * mmScale * chinFactor;
        oy += chinPog * mmScale * 0.2 * chinFactor;
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
    skinTone,
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
    leftScene.background = new THREE.Color("#111622");

    const leftCamera = new THREE.PerspectiveCamera(38, leftWidth / leftHeight, 1, 1000);
    leftCamera.position.set(0, 0, 320);

    const leftRenderer = new THREE.WebGLRenderer({
      canvas: leftCanvasRef.current,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    leftRenderer.outputColorSpace = THREE.SRGBColorSpace;
    leftRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    leftRenderer.toneMappingExposure = 1.18;
    leftRenderer.setSize(leftWidth, leftHeight, false);
    leftRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const leftControls = new OrbitControls(leftCamera, leftRenderer.domElement);
    leftControls.enableDamping = true;
    leftControls.dampingFactor = 0.06;
    leftControls.minDistance = 100;
    leftControls.maxDistance = 600;
    // Cho phép xoay 360 độ tự do quanh khuôn mặt
    leftControls.minPolarAngle = Math.PI / 2 - 0.65;
    leftControls.maxPolarAngle = Math.PI / 2 + 0.65;
    leftControls.enablePan = false;
    leftControls.screenSpacePanning = false;

    const leftGroup = new THREE.Group();
    leftScene.add(leftGroup);

    // Luminous clinical studio lighting: bright ambient + hemisphere bounce + multi-angle key & soft under-fill
    const leftAmbient = new THREE.AmbientLight(0xffffff, 1.05);
    leftScene.add(leftAmbient);
    const leftHemi = new THREE.HemisphereLight(0xfff7ee, 0x1c2436, 0.50);
    leftScene.add(leftHemi);
    const leftKeyLight = new THREE.DirectionalLight(0xffffff, 0.48);
    leftKeyLight.position.set(0.35, 0.25, 1);
    leftCamera.add(leftKeyLight);
    const leftFillLight = new THREE.DirectionalLight(0xffffff, 0.32);
    leftFillLight.position.set(-0.35, -0.15, 1);
    leftCamera.add(leftFillLight);
    leftScene.add(leftCamera);

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
      rightScene.background = new THREE.Color("#111622");

      const rightCamera = new THREE.PerspectiveCamera(38, rightWidth / rightHeight, 1, 1000);
      rightCamera.position.copy(leftCamera.position);

      const rightRenderer = new THREE.WebGLRenderer({
        canvas: rightCanvasRef.current,
        antialias: true,
        preserveDrawingBuffer: true,
      });
      rightRenderer.outputColorSpace = THREE.SRGBColorSpace;
      rightRenderer.toneMapping = THREE.ACESFilmicToneMapping;
      rightRenderer.toneMappingExposure = 1.18;
      rightRenderer.setSize(rightWidth, rightHeight, false);
      rightRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      const rightControls = new OrbitControls(rightCamera, rightRenderer.domElement);
      rightControls.enableDamping = true;
      rightControls.dampingFactor = 0.06;
      rightControls.minDistance = 100;
      rightControls.maxDistance = 600;
      rightControls.minPolarAngle = Math.PI / 2 - 0.65;
      rightControls.maxPolarAngle = Math.PI / 2 + 0.65;
      rightControls.enablePan = false;
      rightControls.screenSpacePanning = false;

      const rightGroup = new THREE.Group();
      rightScene.add(rightGroup);

      // Luminous clinical studio lighting for right viewport
      const rightAmbient = new THREE.AmbientLight(0xffffff, 1.05);
      rightScene.add(rightAmbient);
      const rightHemi = new THREE.HemisphereLight(0xfff7ee, 0x1c2436, 0.50);
      rightScene.add(rightHemi);
      const rightKeyLight = new THREE.DirectionalLight(0xffffff, 0.48);
      rightKeyLight.position.set(0.35, 0.25, 1);
      rightCamera.add(rightKeyLight);
      const rightFillLight = new THREE.DirectionalLight(0xffffff, 0.32);
      rightFillLight.position.set(-0.35, -0.15, 1);
      rightCamera.add(rightFillLight);
      rightScene.add(rightCamera);

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

    // 3. LOAD KHỐI 3D THẬT TỪ SERVER
    const gltfLoader = new GLTFLoader();
    const objLoader = new OBJLoader();

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
      const targetScale = 165 / maxDim;

      // Anatomical facial proportion normalization:
      // Normal human bizygomatic-to-facial-height ratio is ~0.74 - 0.78.
      // If a model is excessively narrow/squeezed (e.g. ratio < 0.75 from portrait crop), gently restore width.
      const rawRatio = size.y > 0 ? (size.x / size.y) : 0.75;
      const widthMultiplier = rawRatio < 0.75 ? Math.min(1.18, 0.76 / Math.max(0.55, rawRatio)) : 1.05;
      const scaleX = targetScale * widthMultiplier;
      const scaleY = targetScale;
      const scaleZ = targetScale;

      modelLeft.scale.set(scaleX, scaleY, scaleZ);
      modelRight.scale.set(scaleX, scaleY, scaleZ);
      modelLeft.position.set(-center.x * scaleX, -center.y * scaleY, -center.z * scaleZ);
      modelRight.position.set(-center.x * scaleX, -center.y * scaleY, -center.z * scaleZ);

      let leftMesh: THREE.Mesh | null = null;
      let rightMesh: THREE.Mesh | null = null;

      const enhanceMesh = (child: THREE.Object3D, isLeft: boolean) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (isLeft && !leftMesh) leftMesh = mesh;
          if (!isLeft && !rightMesh) rightMesh = mesh;
          if (mesh.geometry) {
            mesh.geometry.computeVertexNormals();
          }
          if (mesh.material) {
            const oldMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
            const map = (oldMat as THREE.MeshStandardMaterial).map;
            if (map) {
              map.colorSpace = THREE.SRGBColorSpace;
              map.anisotropy = Math.min(leftRenderer.capabilities.getMaxAnisotropy(), 16);
              map.minFilter = THREE.LinearMipmapLinearFilter;
              map.magFilter = THREE.LinearFilter;
              map.generateMipmaps = true;
              map.needsUpdate = true;
            }
            // Ultra-crisp photorealistic Asian skin PBR shader with natural soft clinical matte finish
            mesh.material = new THREE.MeshStandardMaterial({
              map,
              side: THREE.DoubleSide,
              roughness: 0.84,
              metalness: 0.0,
            });
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

    let isCancelled = false;
    let retryAttempt = 0;
    const maxRetryAttempts = 30; // Chờ tối đa 45 giây cho AI Engine

    const fetchAndLoadModel = async (): Promise<boolean> => {
      if (!patientId || isCancelled) return false;

      const now = Date.now();
      const urls = [
        `/api/patients/${patientId}/model-file?t=${now}&retry=${retryAttempt}`,
        `/models/patients/${patientId}/reconstruction/baseline.glb?t=${now}`,
        `/models/patients/${patientId}/baseline.glb?t=${now}`,
        `/data/patients/${patientId}/model.glb?t=${now}`,
      ];

      for (const url of urls) {
        if (isCancelled) return false;
        try {
          const res = await fetch(url, { cache: "no-store", headers: { Pragma: "no-cache" } });
          if (res.ok && res.status === 200) {
            const buffer = await res.arrayBuffer();
            if (buffer && buffer.byteLength > 500) {
              // Parse GLB binary
              return await new Promise<boolean>((resolve) => {
                gltfLoader.parse(
                  buffer,
                  "",
                  (gltf) => {
                    if (isCancelled) return resolve(false);
                    const modelLeft = cloneModelWithDeepGeometry(gltf.scene);
                    const modelRight = cloneModelWithDeepGeometry(gltf.scene);
                    attachLoadedScene(modelLeft, modelRight);
                    resolve(true);
                  },
                  () => {
                    resolve(false);
                  }
                );
              });
            }
          }
        } catch {
          // Continue to next candidate URL
        }
      }
      return false;
    };

    const startModelPolling = async () => {
      const success = await fetchAndLoadModel();
      if (success || isCancelled) return;

      if (retryAttempt < maxRetryAttempts) {
        retryAttempt++;
        setTimeout(startModelPolling, 1500);
      } else {
        if (!isCancelled) {
          setNoValidBaseline(true);
          setIsLoading(false);
        }
      }
    };

    startModelPolling();

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
      isCancelled = true;
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

  // Reactive real-time Skin Tone & Lighting adjustment (Khử bóng nhờn + chỉnh tông da y khoa trực tiếp)
  useEffect(() => {
    const tone = skinTone ?? { brightness: 1.05, warmth: 0, smoothness: 0.82 };

    const exposure = Math.max(0.85, Math.min(1.40, tone.brightness));
    if (leftSceneRef.current) {
      leftSceneRef.current.renderer.toneMappingExposure = exposure;
    }
    if (rightSceneRef.current) {
      rightSceneRef.current.renderer.toneMappingExposure = exposure;
    }

    let r = 1.0, g = 1.0, b = 1.0;
    if (tone.warmth > 0) {
      // Warm / Pink tone: subtly enhances warm rosy tones on skin
      r = 1.0 + (tone.warmth / 20) * 0.04;
      g = 1.0 - (tone.warmth / 20) * 0.012;
      b = 1.0 - (tone.warmth / 20) * 0.035;
    } else if (tone.warmth < 0) {
      // Cool / Clinical porcelain tone: enhances bright neutral clinical clarity
      r = 1.0 + (tone.warmth / 20) * 0.015;
      g = 1.0 + (tone.warmth / 20) * 0.008;
      b = 1.0 - (tone.warmth / 20) * 0.03;
    }
    const tintColor = new THREE.Color(r, g, b);

    const updateMeshTone = (mesh: THREE.Mesh | null) => {
      if (!mesh || !mesh.material) return;
      const mat = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.MeshStandardMaterial;
      if (mat && mat.isMeshStandardMaterial) {
        mat.color.copy(tintColor);
        mat.roughness = Math.max(0.60, Math.min(0.95, tone.smoothness));
        mat.needsUpdate = true;
      }
    };

    updateMeshTone(leftSceneRef.current?.headMesh ?? null);
    updateMeshTone(rightSceneRef.current?.headMesh ?? null);
  }, [skinTone]);

  useImperativeHandle(ref, () => ({
    captureSnapshot: () => {
      if (rightCanvasRef.current) {
        return rightCanvasRef.current.toDataURL("image/jpeg", 0.95);
      }
      return null;
    },
    resetCamera: () => {
      if (leftSceneRef.current) {
        leftSceneRef.current.camera.position.set(0, 0, 320);
        leftSceneRef.current.controls.target.set(0, 0, 0);
        leftSceneRef.current.controls.update();
      }
      if (rightSceneRef.current) {
        rightSceneRef.current.camera.position.set(0, 0, 320);
        rightSceneRef.current.controls.target.set(0, 0, 0);
        rightSceneRef.current.controls.update();
      }
    },
    resetView: () => {
      if (leftSceneRef.current) {
        leftSceneRef.current.camera.position.set(0, 0, 320);
        leftSceneRef.current.controls.target.set(0, 0, 0);
        leftSceneRef.current.controls.update();
      }
      if (rightSceneRef.current) {
        rightSceneRef.current.camera.position.set(0, 0, 320);
        rightSceneRef.current.controls.target.set(0, 0, 0);
        rightSceneRef.current.controls.update();
      }
    },
    goToAngle: (azimuthDeg: number, polarDeg?: number) => {
      const radius = 320;
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
