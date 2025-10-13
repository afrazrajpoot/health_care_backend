-- CreateTable
CREATE TABLE "IntakeLink" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "patient" TEXT NOT NULL,
    "dob" TIMESTAMP(3) NOT NULL,
    "visitType" TEXT NOT NULL,
    "language" TEXT NOT NULL,
    "mode" TEXT NOT NULL,
    "bodyParts" TEXT NOT NULL,
    "expiresInDays" INTEGER NOT NULL,
    "requireDobVerify" BOOLEAN NOT NULL,
    "appointments" JSONB,
    "medications" JSONB,
    "painLevel" INTEGER,
    "adls" JSONB,
    "therapyFeedback" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "IntakeLink_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "IntakeLink_token_key" ON "IntakeLink"("token");
