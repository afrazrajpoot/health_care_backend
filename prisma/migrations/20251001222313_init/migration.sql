/*
  Warnings:

  - You are about to drop the `WhatsNewSinceLastVisit` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "WhatsNewSinceLastVisit" DROP CONSTRAINT "WhatsNewSinceLastVisit_documentId_fkey";

-- AlterTable
ALTER TABLE "Document" ADD COLUMN     "whatsNew" JSONB;

-- DropTable
DROP TABLE "WhatsNewSinceLastVisit";
