/*
  Warnings:

  - You are about to drop the column `patientQuizId` on the `Document` table. All the data in the column will be lost.
  - You are about to drop the column `documentId` on the `PatientQuiz` table. All the data in the column will be lost.
  - You are about to drop the column `patientId` on the `PatientQuiz` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE "Document" DROP CONSTRAINT "Document_patientQuizId_fkey";

-- DropIndex
DROP INDEX "Document_patientQuizId_key";

-- DropIndex
DROP INDEX "PatientQuiz_documentId_key";

-- AlterTable
ALTER TABLE "Document" DROP COLUMN "patientQuizId";

-- AlterTable
ALTER TABLE "PatientQuiz" DROP COLUMN "documentId",
DROP COLUMN "patientId",
ADD COLUMN     "dob" TEXT,
ADD COLUMN     "doi" TEXT,
ADD COLUMN     "patientName" TEXT;
