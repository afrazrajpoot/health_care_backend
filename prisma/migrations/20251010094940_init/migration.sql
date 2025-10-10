/*
  Warnings:

  - You are about to drop the column `appts` on the `PatientQuiz` table. All the data in the column will be lost.
  - You are about to drop the column `barrier` on the `PatientQuiz` table. All the data in the column will be lost.
  - You are about to drop the column `pain` on the `PatientQuiz` table. All the data in the column will be lost.
  - You are about to drop the column `trend` on the `PatientQuiz` table. All the data in the column will be lost.
  - You are about to drop the column `workAbility` on the `PatientQuiz` table. All the data in the column will be lost.
  - You are about to drop the column `workDiff` on the `PatientQuiz` table. All the data in the column will be lost.
  - The `newAppt` column on the `PatientQuiz` table would be dropped and recreated. This will lead to data loss if there is data in the column.

*/
-- AlterTable
ALTER TABLE "PatientQuiz" DROP COLUMN "appts",
DROP COLUMN "barrier",
DROP COLUMN "pain",
DROP COLUMN "trend",
DROP COLUMN "workAbility",
DROP COLUMN "workDiff",
ADD COLUMN     "refill" JSONB,
DROP COLUMN "newAppt",
ADD COLUMN     "newAppt" JSONB;
