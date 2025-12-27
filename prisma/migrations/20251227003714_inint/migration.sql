-- CreateTable
CREATE TABLE "users" (
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

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "accounts" (
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

    CONSTRAINT "accounts_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "sessions" (
    "id" TEXT NOT NULL,
    "sessionToken" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "expires" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "verification_tokens" (
    "identifier" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "expires" TIMESTAMP(3) NOT NULL
);

-- CreateTable
CREATE TABLE "documents" (
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

    CONSTRAINT "documents_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "summary_snapshots" (
    "id" TEXT NOT NULL,
    "dx" TEXT,
    "keyConcern" TEXT,
    "nextStep" TEXT,
    "bodyPart" TEXT,
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

    CONSTRAINT "summary_snapshots_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "document_summaries" (
    "id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "summary" TEXT NOT NULL,
    "documentId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "document_summaries_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "audit_logs" (
    "id" TEXT NOT NULL,
    "userId" TEXT,
    "email" TEXT,
    "action" TEXT NOT NULL,
    "ipAddress" TEXT,
    "userAgent" TEXT,
    "path" TEXT,
    "method" TEXT,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "audit_logs_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "fail_docs" (
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
    "summary" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "fail_docs_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "tasks" (
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
    "assignee" TEXT DEFAULT 'Unclaimed',
    "type" TEXT DEFAULT 'internal',
    "quickNotes" JSONB DEFAULT '{"status_update": "", "details": "", "one_line_note": ""}',
    "followUpTaskId" TEXT,
    "documentId" TEXT,
    "physicianId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "tasks_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "workflow_stats" (
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

    CONSTRAINT "workflow_stats_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "checkout_sessions" (
    "id" TEXT NOT NULL,
    "stripeSessionId" TEXT NOT NULL,
    "physicianId" TEXT NOT NULL,
    "plan" TEXT NOT NULL,
    "amount" INTEGER NOT NULL,
    "status" TEXT NOT NULL,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "checkout_sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "patient_intake_updates" (
    "id" TEXT NOT NULL,
    "patientName" TEXT NOT NULL,
    "dob" TEXT,
    "claimNumber" TEXT,
    "doi" TEXT,
    "patientQuizId" TEXT,
    "documentId" TEXT,
    "keyPatientReportedChanges" TEXT,
    "systemInterpretation" TEXT,
    "keyFindings" TEXT,
    "adlEffectPoints" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "intakePatientPoints" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "medRefillsRequested" TEXT,
    "newAppointments" TEXT,
    "adlChanges" TEXT,
    "intakeData" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "patient_intake_updates_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "accounts_provider_providerAccountId_key" ON "accounts"("provider", "providerAccountId");

-- CreateIndex
CREATE UNIQUE INDEX "sessions_sessionToken_key" ON "sessions"("sessionToken");

-- CreateIndex
CREATE UNIQUE INDEX "verification_tokens_token_key" ON "verification_tokens"("token");

-- CreateIndex
CREATE UNIQUE INDEX "verification_tokens_identifier_token_key" ON "verification_tokens"("identifier", "token");

-- CreateIndex
CREATE UNIQUE INDEX "documents_fileHash_userId_key" ON "documents"("fileHash", "userId");

-- CreateIndex
CREATE INDEX "body_part_snapshots_mode_idx" ON "body_part_snapshots"("mode");

-- CreateIndex
CREATE INDEX "body_part_snapshots_documentId_idx" ON "body_part_snapshots"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "summary_snapshots_documentId_key" ON "summary_snapshots"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "document_summaries_documentId_key" ON "document_summaries"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "adls_documentId_key" ON "adls"("documentId");

-- CreateIndex
CREATE INDEX "adls_mode_idx" ON "adls"("mode");

-- CreateIndex
CREATE INDEX "adls_documentId_idx" ON "adls"("documentId");

-- CreateIndex
CREATE UNIQUE INDEX "intake_links_token_key" ON "intake_links"("token");

-- CreateIndex
CREATE INDEX "tasks_type_idx" ON "tasks"("type");

-- CreateIndex
CREATE UNIQUE INDEX "checkout_sessions_stripeSessionId_key" ON "checkout_sessions"("stripeSessionId");

-- CreateIndex
CREATE INDEX "checkout_sessions_stripeSessionId_idx" ON "checkout_sessions"("stripeSessionId");

-- CreateIndex
CREATE INDEX "checkout_sessions_physicianId_idx" ON "checkout_sessions"("physicianId");

-- CreateIndex
CREATE INDEX "patient_intake_updates_patientName_dob_claimNumber_idx" ON "patient_intake_updates"("patientName", "dob", "claimNumber");

-- CreateIndex
CREATE INDEX "patient_intake_updates_documentId_idx" ON "patient_intake_updates"("documentId");

-- CreateIndex
CREATE INDEX "patient_intake_updates_createdAt_idx" ON "patient_intake_updates"("createdAt");

-- AddForeignKey
ALTER TABLE "accounts" ADD CONSTRAINT "accounts_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "documents" ADD CONSTRAINT "documents_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "body_part_snapshots" ADD CONSTRAINT "body_part_snapshots_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "summary_snapshots" ADD CONSTRAINT "summary_snapshots_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "document_summaries" ADD CONSTRAINT "document_summaries_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "adls" ADD CONSTRAINT "adls_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "tasks" ADD CONSTRAINT "tasks_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "patient_intake_updates" ADD CONSTRAINT "patient_intake_updates_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;
