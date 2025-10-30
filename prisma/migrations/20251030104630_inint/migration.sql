/*
  Warnings:

  - You are about to drop the `BodyPartSnapshot` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "BodyPartSnapshot" DROP CONSTRAINT "BodyPartSnapshot_documentId_fkey";

-- DropTable
DROP TABLE "BodyPartSnapshot";

-- CreateTable
CREATE TABLE "body_part_snapshots" (
    "id" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,
    "bodyPart" TEXT NOT NULL,
    "dx" TEXT NOT NULL,
    "keyConcern" TEXT NOT NULL,
    "nextStep" TEXT,
    "urDecision" TEXT,
    "recommended" TEXT,
    "aiOutcome" TEXT,
    "consultingDoctor" TEXT,
    "keyFindings" TEXT,
    "treatmentApproach" TEXT,
    "clinicalSummary" TEXT,
    "referralDoctor" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "body_part_snapshots_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "body_part_snapshots" ADD CONSTRAINT "body_part_snapshots_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE CASCADE ON UPDATE CASCADE;
