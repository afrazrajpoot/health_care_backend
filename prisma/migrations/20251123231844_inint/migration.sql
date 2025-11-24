/*
  Warnings:

  - You are about to drop the `AuditLog` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `FailDocs` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `intake_links` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `patient_quizzes` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "DocumentSummary" DROP CONSTRAINT "DocumentSummary_documentId_fkey";

-- DropForeignKey
ALTER TABLE "SummarySnapshot" DROP CONSTRAINT "SummarySnapshot_documentId_fkey";

-- DropForeignKey
ALTER TABLE "Task" DROP CONSTRAINT "Task_documentId_fkey";

-- DropTable
DROP TABLE "AuditLog";

-- DropTable
DROP TABLE "FailDocs";

-- DropTable
DROP TABLE "intake_links";

-- DropTable
DROP TABLE "patient_quizzes";

-- AddForeignKey
ALTER TABLE "SummarySnapshot" ADD CONSTRAINT "SummarySnapshot_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DocumentSummary" ADD CONSTRAINT "DocumentSummary_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE CASCADE ON UPDATE CASCADE;
