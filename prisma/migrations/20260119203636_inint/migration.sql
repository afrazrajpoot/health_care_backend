-- DropIndex
DROP INDEX "treatment_histories_patientName_dob_claimNumber_physicianId_key";

-- AlterTable
ALTER TABLE "patient_intake_updates" ADD COLUMN     "generatedPoints" TEXT[] DEFAULT ARRAY[]::TEXT[];
