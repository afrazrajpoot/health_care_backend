/*
  Warnings:

  - You are about to drop the `alerts` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `documents` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "alerts" DROP CONSTRAINT "alerts_documentId_fkey";

-- DropTable
DROP TABLE "alerts";

-- DropTable
DROP TABLE "documents";

-- CreateTable
CREATE TABLE "Document" (
    "id" TEXT NOT NULL,
    "dob" TIMESTAMP(3) NOT NULL,
    "doi" TIMESTAMP(3) NOT NULL,
    "patientName" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "claimNumber" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Document_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SummarySnapshot" (
    "id" TEXT NOT NULL,
    "dx" TEXT NOT NULL,
    "keyConcern" TEXT NOT NULL,
    "nextStep" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,

    CONSTRAINT "SummarySnapshot_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "WhatsNewSinceLastVisit" (
    "id" TEXT NOT NULL,
    "diagnostic" TEXT NOT NULL,
    "qme" TEXT NOT NULL,
    "urDecision" TEXT NOT NULL,
    "legal" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,

    CONSTRAINT "WhatsNewSinceLastVisit_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ADL" (
    "id" TEXT NOT NULL,
    "adlsAffected" TEXT NOT NULL,
    "workRestrictions" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,

    CONSTRAINT "ADL_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DocumentSummary" (
    "id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "summary" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,

    CONSTRAINT "DocumentSummary_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Document_claimNumber_key" ON "Document"("claimNumber");

-- CreateIndex
CREATE UNIQUE INDEX "SummarySnapshot_documentId_key" ON "SummarySnapshot"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "WhatsNewSinceLastVisit_documentId_key" ON "WhatsNewSinceLastVisit"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "ADL_documentId_key" ON "ADL"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "DocumentSummary_documentId_key" ON "DocumentSummary"("documentId");

-- AddForeignKey
ALTER TABLE "SummarySnapshot" ADD CONSTRAINT "SummarySnapshot_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "WhatsNewSinceLastVisit" ADD CONSTRAINT "WhatsNewSinceLastVisit_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ADL" ADD CONSTRAINT "ADL_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DocumentSummary" ADD CONSTRAINT "DocumentSummary_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
