-- CreateTable
CREATE TABLE "patient_intake_updates" (
    "id" TEXT NOT NULL,
    "patientName" TEXT NOT NULL,
    "dob" TEXT,
    "claimNumber" TEXT,
    "doi" TEXT,
    "patientQuizId" TEXT,
    "documentId" TEXT,
    "keyPatientReportedChanges" TEXT,
    "systemInterpretation" TEXT,
    "medRefillsRequested" TEXT,
    "newAppointments" TEXT,
    "adlChanges" TEXT,
    "intakeData" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "patient_intake_updates_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "patient_intake_updates_patientName_dob_claimNumber_idx" ON "patient_intake_updates"("patientName", "dob", "claimNumber");

-- CreateIndex
CREATE INDEX "patient_intake_updates_documentId_idx" ON "patient_intake_updates"("documentId");

-- CreateIndex
CREATE INDEX "patient_intake_updates_createdAt_idx" ON "patient_intake_updates"("createdAt");

-- AddForeignKey
ALTER TABLE "patient_intake_updates" ADD CONSTRAINT "patient_intake_updates_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;
