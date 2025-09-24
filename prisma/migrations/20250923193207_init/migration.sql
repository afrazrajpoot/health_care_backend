/*
  Warnings:

  - You are about to drop the column `department` on the `User` table. All the data in the column will be lost.
  - You are about to drop the column `employeeId` on the `User` table. All the data in the column will be lost.
  - You are about to drop the column `hrId` on the `User` table. All the data in the column will be lost.
  - You are about to drop the column `position` on the `User` table. All the data in the column will be lost.
  - You are about to drop the column `salary` on the `User` table. All the data in the column will be lost.
  - You are about to drop the `AiCareerRecommendation` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Application` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Company` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Department` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Employee` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `IndividualEmployeeReport` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Job` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Notification` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `chat_conversations` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "Application" DROP CONSTRAINT "Application_candidateId_fkey";

-- DropForeignKey
ALTER TABLE "Application" DROP CONSTRAINT "Application_jobId_fkey";

-- DropForeignKey
ALTER TABLE "Job" DROP CONSTRAINT "Job_recruiterId_fkey";

-- DropForeignKey
ALTER TABLE "User" DROP CONSTRAINT "User_employeeId_fkey";

-- DropIndex
DROP INDEX "User_employeeId_key";

-- AlterTable
ALTER TABLE "User" DROP COLUMN "department",
DROP COLUMN "employeeId",
DROP COLUMN "hrId",
DROP COLUMN "position",
DROP COLUMN "salary";

-- DropTable
DROP TABLE "AiCareerRecommendation";

-- DropTable
DROP TABLE "Application";

-- DropTable
DROP TABLE "Company";

-- DropTable
DROP TABLE "Department";

-- DropTable
DROP TABLE "Employee";

-- DropTable
DROP TABLE "IndividualEmployeeReport";

-- DropTable
DROP TABLE "Job";

-- DropTable
DROP TABLE "Notification";

-- DropTable
DROP TABLE "chat_conversations";

-- DropEnum
DROP TYPE "JobStatus";

-- DropEnum
DROP TYPE "JobType";

-- DropEnum
DROP TYPE "UserRole";
