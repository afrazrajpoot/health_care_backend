-- AlterTable
ALTER TABLE "patient_intake_updates" ADD COLUMN     "adlEffectPoints" TEXT[] DEFAULT ARRAY[]::TEXT[],
ADD COLUMN     "intakePatientPoints" TEXT[] DEFAULT ARRAY[]::TEXT[];
