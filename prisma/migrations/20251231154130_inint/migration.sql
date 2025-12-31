/*
  Warnings:

  - You are about to drop the column `summary` on the `fail_docs` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "fail_docs" DROP COLUMN "summary",
ADD COLUMN     "aiSummarizerText" TEXT;
