/*
  Warnings:

  - You are about to drop the column `blobPath` on the `FailDocs` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "FailDocs" DROP COLUMN "blobPath",
ADD COLUMN     "claimNumber" TEXT,
ADD COLUMN     "db" TEXT,
ADD COLUMN     "documentText" TEXT,
ADD COLUMN     "doi" TEXT,
ADD COLUMN     "patientName" TEXT;
