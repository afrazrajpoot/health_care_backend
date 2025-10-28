/*
  Warnings:

  - You are about to drop the column `consultingDoctors` on the `SummarySnapshot` table. All the data in the column will be lost.
  - Added the required column `bodyPart` to the `SummarySnapshot` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "SummarySnapshot" DROP COLUMN "consultingDoctors",
ADD COLUMN     "bodyPart" TEXT NOT NULL,
ADD COLUMN     "consultingDoctor" TEXT[];
