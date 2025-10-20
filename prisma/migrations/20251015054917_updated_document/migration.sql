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
    "physicianId" TEXT,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "Document" (
    "id" TEXT NOT NULL,
    "dob" TEXT,
    "doi" TEXT NOT NULL,
    "patientName" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "claimNumber" TEXT NOT NULL,
    "gcsFileLink" TEXT NOT NULL,
    "briefSummary" TEXT,
    "whatsNew" JSONB,
    "blobPath" TEXT,
    "fileName" TEXT,
    "fileHash" VARCHAR(64),
    "reportDate" TIMESTAMP(3),
    "physicianId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "userId" TEXT,

    CONSTRAINT "Document_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SummarySnapshot" (
    "id" TEXT NOT NULL,
    "dx" TEXT NOT NULL,
    "keyConcern" TEXT NOT NULL,
    "nextStep" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,

    CONSTRAINT "SummarySnapshot_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ADL" (
    "id" TEXT NOT NULL,
    "adlsAffected" TEXT NOT NULL,
    "workRestrictions" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,

    CONSTRAINT "ADL_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DocumentSummary" (
    "id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "summary" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,

    CONSTRAINT "DocumentSummary_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AuditLog" (
    "id" TEXT NOT NULL,
    "userId" TEXT,
    "email" TEXT,
    "action" TEXT NOT NULL,
    "ipAddress" TEXT,
    "userAgent" TEXT,
    "path" TEXT,
    "method" TEXT,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AuditLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PatientQuiz" (
    "id" TEXT NOT NULL,
    "patientName" TEXT,
    "dob" TEXT,
    "doi" TEXT,
    "lang" TEXT NOT NULL,
    "newAppt" JSONB,
    "refill" JSONB,
    "adl" JSONB NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "PatientQuiz_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FailDocs" (
    "id" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "blobPath" TEXT NOT NULL,
    "physicianId" TEXT,

    CONSTRAINT "FailDocs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "intake_links" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "patientName" TEXT NOT NULL,
    "dateOfBirth" TIMESTAMP(3) NOT NULL,
    "visitType" TEXT NOT NULL DEFAULT 'Follow-up',
    "language" TEXT NOT NULL DEFAULT 'en',
    "mode" TEXT NOT NULL DEFAULT 'tele',
    "bodyParts" TEXT,
    "expiresInDays" INTEGER NOT NULL DEFAULT 7,
    "requireAuth" BOOLEAN NOT NULL DEFAULT true,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "intake_links_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Task" (
    "id" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "department" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'Pending',
    "dueDate" TIMESTAMP(3),
    "patient" TEXT NOT NULL,
    "actions" TEXT[] DEFAULT ARRAY['Claim', 'Complete']::TEXT[],
    "sourceDocument" TEXT,
    "quickNotes" JSONB DEFAULT '{"status_update": "", "details": "", "one_line_note": ""}',
    "documentId" TEXT,
    "physicianId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Task_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "Account_provider_providerAccountId_key" ON "Account"("provider", "providerAccountId");

-- CreateIndex
CREATE UNIQUE INDEX "Session_sessionToken_key" ON "Session"("sessionToken");

-- CreateIndex
CREATE UNIQUE INDEX "VerificationToken_token_key" ON "VerificationToken"("token");

-- CreateIndex
CREATE UNIQUE INDEX "VerificationToken_identifier_token_key" ON "VerificationToken"("identifier", "token");

-- CreateIndex
CREATE UNIQUE INDEX "Document_fileHash_userId_key" ON "Document"("fileHash", "userId");

-- CreateIndex
CREATE UNIQUE INDEX "SummarySnapshot_documentId_key" ON "SummarySnapshot"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "ADL_documentId_key" ON "ADL"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "DocumentSummary_documentId_key" ON "DocumentSummary"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "intake_links_token_key" ON "intake_links"("token");

-- AddForeignKey
ALTER TABLE "Account" ADD CONSTRAINT "Account_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Session" ADD CONSTRAINT "Session_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Document" ADD CONSTRAINT "Document_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SummarySnapshot" ADD CONSTRAINT "SummarySnapshot_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ADL" ADD CONSTRAINT "ADL_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DocumentSummary" ADD CONSTRAINT "DocumentSummary_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE SET NULL ON UPDATE CASCADE;
