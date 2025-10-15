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
