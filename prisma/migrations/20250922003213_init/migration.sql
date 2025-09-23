-- CreateEnum
CREATE TYPE "JobType" AS ENUM ('FULL_TIME', 'PART_TIME', 'CONTRACT', 'INTERNSHIP', 'FREELANCE');

-- CreateEnum
CREATE TYPE "JobStatus" AS ENUM ('OPEN', 'CLOSED', 'DRAFT');

-- CreateEnum
CREATE TYPE "UserRole" AS ENUM ('CANDIDATE', 'RECRUITER', 'ADMIN');

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "firstName" TEXT,
    "lastName" TEXT,
    "phoneNumber" TEXT,
    "email" TEXT,
    "password" TEXT,
    "role" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "emailVerified" TIMESTAMP(3),
    "image" TEXT,
    "position" TEXT[],
    "department" TEXT[],
    "hrId" TEXT,
    "salary" TEXT,
    "employeeId" TEXT,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Department" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "hrId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "ingoing" JSONB[],
    "outgoing" JSONB[],
    "promotion" TEXT,
    "transfer" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Department_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Account" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "providerAccountId" TEXT NOT NULL,
    "refresh_token" TEXT,
    "access_token" TEXT,
    "expires_at" INTEGER,
    "refresh_expires_at" INTEGER,
    "token_type" TEXT,
    "scope" TEXT,
    "id_token" TEXT,
    "session_state" TEXT,

    CONSTRAINT "Account_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Session" (
    "id" TEXT NOT NULL,
    "sessionToken" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "expires" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Session_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "VerificationToken" (
    "identifier" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "expires" TIMESTAMP(3) NOT NULL
);

-- CreateTable
CREATE TABLE "Company" (
    "id" TEXT NOT NULL,
    "companyDetail" JSONB NOT NULL,
    "hrId" TEXT[],
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Company_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Employee" (
    "id" TEXT NOT NULL,
    "employeeId" TEXT NOT NULL,
    "firstName" TEXT NOT NULL,
    "lastName" TEXT NOT NULL,
    "address" TEXT,
    "dateOfBirth" TIMESTAMP(3),
    "hireDate" TIMESTAMP(3),
    "manager" TEXT,
    "bio" TEXT,
    "avatar" TEXT,
    "skills" JSONB,
    "education" JSONB,
    "experience" JSONB,
    "resume" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Employee_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "IndividualEmployeeReport" (
    "id" SERIAL NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "userId" TEXT NOT NULL,
    "executiveSummary" TEXT NOT NULL,
    "hrId" TEXT,
    "departement" TEXT,
    "geniusFactorScore" INTEGER NOT NULL,
    "geniusFactorProfileJson" JSONB NOT NULL,
    "currentRoleAlignmentAnalysisJson" JSONB NOT NULL,
    "internalCareerOpportunitiesJson" JSONB NOT NULL,
    "retentionAndMobilityStrategiesJson" JSONB NOT NULL,
    "developmentActionPlanJson" JSONB NOT NULL,
    "personalizedResourcesJson" JSONB NOT NULL,
    "dataSourcesAndMethodologyJson" JSONB NOT NULL,
    "risk_analysis" JSONB NOT NULL,

    CONSTRAINT "IndividualEmployeeReport_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Job" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "location" TEXT,
    "salary" INTEGER,
    "type" "JobType" NOT NULL DEFAULT 'FULL_TIME',
    "status" "JobStatus" NOT NULL DEFAULT 'OPEN',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "recruiterId" TEXT NOT NULL,

    CONSTRAINT "Job_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Application" (
    "id" TEXT NOT NULL,
    "coverLetter" TEXT,
    "resumeUrl" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "jobId" TEXT NOT NULL,
    "candidateId" TEXT NOT NULL,

    CONSTRAINT "Application_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AiCareerRecommendation" (
    "id" TEXT NOT NULL,
    "employeeId" TEXT NOT NULL,
    "careerRecommendation" TEXT NOT NULL,

    CONSTRAINT "AiCareerRecommendation_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Notification" (
    "id" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "hrId" TEXT NOT NULL,
    "employeeName" TEXT NOT NULL,
    "employeeEmail" TEXT NOT NULL,
    "employeeId" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Notification_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "chat_conversations" (
    "id" TEXT NOT NULL,
    "hr_id" TEXT NOT NULL,
    "department" TEXT NOT NULL,
    "messages" JSONB NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "chat_conversations_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "User_employeeId_key" ON "User"("employeeId");

-- CreateIndex
CREATE UNIQUE INDEX "Account_provider_providerAccountId_key" ON "Account"("provider", "providerAccountId");

-- CreateIndex
CREATE UNIQUE INDEX "Session_sessionToken_key" ON "Session"("sessionToken");

-- CreateIndex
CREATE UNIQUE INDEX "VerificationToken_token_key" ON "VerificationToken"("token");

-- CreateIndex
CREATE UNIQUE INDEX "VerificationToken_identifier_token_key" ON "VerificationToken"("identifier", "token");

-- CreateIndex
CREATE UNIQUE INDEX "Employee_employeeId_key" ON "Employee"("employeeId");

-- CreateIndex
CREATE UNIQUE INDEX "chat_conversations_hr_id_department_key" ON "chat_conversations"("hr_id", "department");

-- AddForeignKey
ALTER TABLE "User" ADD CONSTRAINT "User_employeeId_fkey" FOREIGN KEY ("employeeId") REFERENCES "Employee"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Account" ADD CONSTRAINT "Account_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Session" ADD CONSTRAINT "Session_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Job" ADD CONSTRAINT "Job_recruiterId_fkey" FOREIGN KEY ("recruiterId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Application" ADD CONSTRAINT "Application_jobId_fkey" FOREIGN KEY ("jobId") REFERENCES "Job"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Application" ADD CONSTRAINT "Application_candidateId_fkey" FOREIGN KEY ("candidateId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
