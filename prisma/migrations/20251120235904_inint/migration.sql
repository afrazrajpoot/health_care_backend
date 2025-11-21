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
    "mode" TEXT DEFAULT 'wc',
    "reportDate" TIMESTAMP(3),
    "originalName" TEXT,
    "physicianId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "ur_denial_reason" TEXT,
    "userId" TEXT,

    CONSTRAINT "Document_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "body_part_snapshots" (
    "id" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,
    "mode" TEXT NOT NULL DEFAULT 'wc',
    "bodyPart" TEXT,
    "condition" TEXT,
    "conditionSeverity" TEXT,
    "symptoms" TEXT,
    "medications" TEXT,
    "chronicCondition" BOOLEAN NOT NULL DEFAULT false,
    "comorbidities" TEXT,
    "lifestyleRecommendations" TEXT,
    "injuryType" TEXT,
    "workRelatedness" TEXT,
    "permanentImpairment" TEXT,
    "mmiStatus" TEXT,
    "returnToWorkPlan" TEXT,
    "dx" TEXT NOT NULL,
    "keyConcern" TEXT NOT NULL,
    "nextStep" TEXT,
    "urDecision" TEXT,
    "recommended" TEXT,
    "aiOutcome" TEXT,
    "consultingDoctor" TEXT,
    "keyFindings" TEXT,
    "treatmentApproach" TEXT,
    "clinicalSummary" TEXT,
    "referralDoctor" TEXT,
    "adlsAffected" TEXT,
    "painLevel" TEXT,
    "functionalLimitations" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "body_part_snapshots_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SummarySnapshot" (
    "id" TEXT NOT NULL,
    "dx" TEXT NOT NULL,
    "keyConcern" TEXT NOT NULL,
    "nextStep" TEXT NOT NULL,
    "bodyPart" TEXT NOT NULL,
    "urDecision" TEXT,
    "recommended" TEXT,
    "aiOutcome" TEXT,
    "consultingDoctor" TEXT,
    "keyFindings" TEXT,
    "treatmentApproach" TEXT,
    "clinicalSummary" TEXT,
    "referralDoctor" TEXT,
    "documentId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SummarySnapshot_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DocumentSummary" (
    "id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "summary" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "DocumentSummary_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "adls" (
    "id" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,
    "mode" TEXT NOT NULL DEFAULT 'wc',
    "adlsAffected" TEXT NOT NULL,
    "workRestrictions" TEXT NOT NULL,
    "dailyLivingImpact" TEXT,
    "functionalLimitations" TEXT,
    "symptomImpact" TEXT,
    "qualityOfLife" TEXT,
    "workImpact" TEXT,
    "physicalDemands" TEXT,
    "workCapacity" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "adls_pkey" PRIMARY KEY ("id")
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
    "claimNumber" TEXT,

    CONSTRAINT "patient_quizzes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FailDocs" (
    "id" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "physicianId" TEXT,
    "claimNumber" TEXT,
    "documentText" TEXT,
    "doi" TEXT,
    "patientName" TEXT,
    "blobPath" TEXT,
    "fileHash" TEXT,
    "fileName" TEXT,
    "gcsFileLink" TEXT,
    "dob" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

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
    "claimNumber" TEXT,

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
    "reason" TEXT,
    "actions" TEXT[] DEFAULT ARRAY['Claim', 'Complete']::TEXT[],
    "sourceDocument" TEXT,
    "claimNumber" TEXT,
    "quickNotes" JSONB DEFAULT '{"status_update": "", "details": "", "one_line_note": ""}',
    "followUpTaskId" TEXT,
    "documentId" TEXT,
    "physicianId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Task_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "WorkflowStats" (
    "id" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "referralsProcessed" INTEGER NOT NULL DEFAULT 0,
    "rfasMonitored" INTEGER NOT NULL DEFAULT 0,
    "qmeUpcoming" INTEGER NOT NULL DEFAULT 0,
    "payerDisputes" INTEGER NOT NULL DEFAULT 0,
    "externalDocs" INTEGER NOT NULL DEFAULT 0,
    "intakes_created" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WorkflowStats_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "subscriptions" (
    "id" TEXT NOT NULL,
    "physicianId" TEXT NOT NULL,
    "plan" TEXT NOT NULL,
    "amountTotal" INTEGER NOT NULL,
    "status" TEXT NOT NULL,
    "stripeCustomerId" TEXT,
    "stripeSubscriptionId" TEXT,
    "documentParse" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "subscriptions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CheckoutSession" (
    "id" TEXT NOT NULL,
    "stripeSessionId" TEXT NOT NULL,
    "physicianId" TEXT NOT NULL,
    "plan" TEXT NOT NULL,
    "amount" INTEGER NOT NULL,
    "status" TEXT NOT NULL,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "CheckoutSession_pkey" PRIMARY KEY ("id")
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
CREATE INDEX "body_part_snapshots_mode_idx" ON "body_part_snapshots"("mode");

-- CreateIndex
CREATE INDEX "body_part_snapshots_documentId_idx" ON "body_part_snapshots"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "SummarySnapshot_documentId_key" ON "SummarySnapshot"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "DocumentSummary_documentId_key" ON "DocumentSummary"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "adls_documentId_key" ON "adls"("documentId");

-- CreateIndex
CREATE INDEX "adls_mode_idx" ON "adls"("mode");

-- CreateIndex
CREATE INDEX "adls_documentId_idx" ON "adls"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "intake_links_token_key" ON "intake_links"("token");

-- CreateIndex
CREATE UNIQUE INDEX "CheckoutSession_stripeSessionId_key" ON "CheckoutSession"("stripeSessionId");

-- CreateIndex
CREATE INDEX "CheckoutSession_stripeSessionId_idx" ON "CheckoutSession"("stripeSessionId");

-- CreateIndex
CREATE INDEX "CheckoutSession_physicianId_idx" ON "CheckoutSession"("physicianId");

-- AddForeignKey
ALTER TABLE "Account" ADD CONSTRAINT "Account_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Session" ADD CONSTRAINT "Session_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Document" ADD CONSTRAINT "Document_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "body_part_snapshots" ADD CONSTRAINT "body_part_snapshots_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SummarySnapshot" ADD CONSTRAINT "SummarySnapshot_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DocumentSummary" ADD CONSTRAINT "DocumentSummary_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "adls" ADD CONSTRAINT "adls_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Task" ADD CONSTRAINT "Task_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "Document"("id") ON DELETE SET NULL ON UPDATE CASCADE;
