-- AlterTable
ALTER TABLE "SummarySnapshot" ALTER COLUMN "consultingDoctor" DROP NOT NULL,
ALTER COLUMN "consultingDoctor" SET DATA TYPE TEXT;
