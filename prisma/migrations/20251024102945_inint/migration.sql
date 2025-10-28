-- AlterTable
ALTER TABLE "SummarySnapshot" ADD COLUMN     "aiOutcome" TEXT,
ADD COLUMN     "consultingDoctors" TEXT[],
ADD COLUMN     "recommended" TEXT,
ADD COLUMN     "urDecision" TEXT;
