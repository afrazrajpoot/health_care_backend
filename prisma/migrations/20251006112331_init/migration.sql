-- CreateTable
CREATE TABLE "PatientQuiz" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "lang" TEXT NOT NULL,
    "newAppt" TEXT NOT NULL,
    "appts" JSONB,
    "pain" INTEGER NOT NULL,
    "workDiff" TEXT NOT NULL,
    "trend" TEXT NOT NULL,
    "workAbility" TEXT NOT NULL,
    "barrier" TEXT NOT NULL,
    "adl" JSONB NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "PatientQuiz_pkey" PRIMARY KEY ("id")
);
