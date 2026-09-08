import { notFound } from "next/navigation";
import GuidedFaceScan from "@/components/scan/GuidedFaceScan";
import { getPatient } from "@/lib/db";

export default async function PatientScanPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const patient = await getPatient(id);
  if (!patient) notFound();
  return <GuidedFaceScan patientId={patient.id} />;
}
