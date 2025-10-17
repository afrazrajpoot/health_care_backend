/*
  Warnings:

  - You are about to drop the `PatientQuiz` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropTable
DROP TABLE "PatientQuiz";

-- CreateTable
CREATE TABLE "patient_quizzes" (
    "id" TEXT NOT NULL,
    "patientName" TEXT NOT NULL,
    "dob" TEXT,
    "doi" TEXT,
    "lang" TEXT NOT NULL,
    "bodyAreas" TEXT,
    "newAppointments" JSONB,
    "refill" JSONB,
    "adl" JSONB NOT NULL,
    "therapies" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "patient_quizzes_pkey" PRIMARY KEY ("id")
);
