-- CreateTable
CREATE TABLE "documents" (
    "id" TEXT NOT NULL,
    "originalName" TEXT NOT NULL,
    "fileSize" INTEGER NOT NULL,
    "mimeType" TEXT NOT NULL,
    "extractedText" TEXT NOT NULL,
    "pages" INTEGER NOT NULL DEFAULT 0,
    "confidence" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "entities" JSONB[],
    "tables" JSONB[],
    "formFields" JSONB[],
    "patientName" TEXT,
    "patientEmail" TEXT,
    "claimNumber" TEXT,
    "reportTitle" TEXT,
    "reportDate" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'normal',
    "summary" TEXT[],
    "originalReport" TEXT NOT NULL,
    "processingTimeMs" INTEGER NOT NULL DEFAULT 0,
    "analysisSuccess" BOOLEAN NOT NULL DEFAULT true,
    "errorMessage" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "documents_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "alerts" (
    "id" TEXT NOT NULL,
    "alertType" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "description" TEXT,
    "documentId" TEXT NOT NULL,
    "isResolved" BOOLEAN NOT NULL DEFAULT false,
    "resolvedAt" TIMESTAMP(3),
    "resolvedBy" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "alerts_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "alerts" ADD CONSTRAINT "alerts_documentId_fkey" FOREIGN KEY ("documentId") REFERENCES "documents"("id") ON DELETE CASCADE ON UPDATE CASCADE;
