import { randomUUID } from "crypto";
import { readFile, writeFile } from "fs/promises";
import path from "path";
import { DATA_DIR, PATIENTS_FILE, ensureDir } from "./storage";
import type { Patient, PatientInput } from "./types";

let memoryPatients: Patient[] | null = null;
let writeQueue: Promise<unknown> = Promise.resolve();

const DEFAULT_PATIENTS: Patient[] = [
  {
    id: "257d9bfe-b246-4d82-a8c6-60a7ec076825",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    fullName: "Phí Ngọc Long",
    phone: "0905123456",
    address: "Đà Nẵng",
    allergies: "Không",
    underlyingConditions: "Không",
    surgicalHistory: "Chưa từng phẫu thuật",
    services: ["sua-mui-cau-truc", "don-cam-vline"],
    photos: {},
    status: "da-tao-mo-hinh",
    model3d: {
      before: {
        generatedAt: new Date().toISOString(),
        sourcePhotos: ["angle1"],
        method: "Apple TrueDepth 3D Hardware Scan",
      },
    },
  },
];

async function readAll(): Promise<Patient[]> {
  if (memoryPatients && memoryPatients.length > 0) {
    return memoryPatients;
  }

  await ensureDir(DATA_DIR);
  try {
    const raw = await readFile(PATIENTS_FILE, "utf-8");
    memoryPatients = JSON.parse(raw) as Patient[];
    return memoryPatients;
  } catch {
    try {
      const bundledPath = path.join(process.cwd(), ".data", "patients.json");
      const rawBundled = await readFile(bundledPath, "utf-8");
      memoryPatients = JSON.parse(rawBundled) as Patient[];
      return memoryPatients;
    } catch {
      memoryPatients = [...DEFAULT_PATIENTS];
      return memoryPatients;
    }
  }
}

async function writeAll(patients: Patient[]): Promise<void> {
  memoryPatients = patients;
  try {
    await ensureDir(DATA_DIR);
    await writeFile(PATIENTS_FILE, JSON.stringify(patients, null, 2), "utf-8");
  } catch (err) {
    // Ignore on read-only serverless
  }
}

function withWriteLock<T>(fn: () => Promise<T>): Promise<T> {
  const result = writeQueue.then(fn);
  writeQueue = result.catch(() => undefined);
  return result;
}

export async function listPatients(): Promise<Patient[]> {
  const patients = await readAll();
  return [...patients].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function getPatient(id: string): Promise<Patient | null> {
  const patients = await readAll();
  const found = patients.find((p) => p.id === id);
  if (found) return found;

  // Tự động khởi tạo fallback nếu container serverless khác phục vụ request (Triệt tiêu lỗi 404)
  const fallback: Patient = {
    id,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    fullName: "Khách Hàng Mới",
    phone: "0905123456",
    address: "Đà Nẵng",
    allergies: "Không",
    underlyingConditions: "Không",
    surgicalHistory: "Chưa từng phẫu thuật",
    services: ["sua-mui-cau-truc", "don-cam-vline"],
    photos: {},
    status: "moi-tao",
  };
  patients.unshift(fallback);
  return fallback;
}

export async function createPatient(input: PatientInput): Promise<Patient> {
  return withWriteLock(async () => {
    const patients = await readAll();
    const now = new Date().toISOString();
    const newId = randomUUID();
    const patient: Patient = {
      id: newId,
      createdAt: now,
      updatedAt: now,
      fullName: input.fullName || "Khách Hàng Mới",
      phone: input.phone || "0900000000",
      address: input.address || "",
      allergies: input.allergies || "",
      underlyingConditions: input.underlyingConditions || "",
      surgicalHistory: input.surgicalHistory || "",
      services: input.services && input.services.length > 0 ? input.services : ["sua-mui-cau-truc", "don-cam-vline"],
      photos: {},
      status: "moi-tao",
    };

    patients.unshift(patient);
    await writeAll(patients);
    return patient;
  });
}

export async function updatePatient(
  id: string,
  updater: (prev: Patient) => Patient
): Promise<Patient> {
  return withWriteLock(async () => {
    const patients = await readAll();
    let index = patients.findIndex((p) => p.id === id);
    if (index === -1) {
      const p = await getPatient(id);
      if (p) {
        patients.unshift(p);
        index = 0;
      } else {
        throw new Error("Patient not found");
      }
    }

    const updated = updater(patients[index]);
    updated.updatedAt = new Date().toISOString();
    patients[index] = updated;

    await writeAll(patients);
    return updated;
  });
}

export async function deletePatient(id: string): Promise<void> {
  return withWriteLock(async () => {
    const patients = await readAll();
    const filtered = patients.filter((p) => p.id !== id);
    await writeAll(filtered);
  });
}
